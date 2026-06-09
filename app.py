import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from scoreboard_parser import parse_scoreboard


# ============================================================
# SETTINGS
# ============================================================

MAX_UPLOAD_SIZE_MB = 10
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}

# Set API_KEY in Railway Variables before sharing the public endpoint.
# When API_KEY is empty, authentication is disabled for initial testing.
EXPECTED_API_KEY = os.getenv(
    "API_KEY",
    ""
).strip()

# EasyOCR is memory-intensive. Process only one screenshot at a time
# to reduce the likelihood of exceeding Railway Free Trial RAM limits.
OCR_SEMAPHORE = asyncio.Semaphore(1)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Scoreboard OCR API",
    description=(
        "Extracts structured placement, team, player IGN, "
        "kill-count, confidence, and review-warning data "
        "from scoreboard screenshots."
    ),
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


# ============================================================
# HELPERS
# ============================================================

def validate_api_key(
    supplied_api_key: str | None
) -> None:
    """
    Require an API key only where the API_KEY environment variable
    has been configured in Railway.
    """

    if not EXPECTED_API_KEY:
        return

    if supplied_api_key != EXPECTED_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key."
        )


def validate_extension(
    filename: str
) -> str:
    """
    Confirm that the uploaded file has a supported image extension.
    """

    extension = Path(
        filename
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image type. "
                "Upload a JPG, JPEG, PNG, or WEBP image."
            )
        )

    return extension


async def save_upload_to_temporary_file(
    uploaded_file: UploadFile,
    extension: str
) -> str:
    """
    Store an uploaded image temporarily and return its local path.
    """

    image_bytes = await uploaded_file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty."
        )

    if len(image_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"The uploaded image exceeds the "
                f"{MAX_UPLOAD_SIZE_MB} MB limit."
            )
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=extension
    ) as temporary_file:
        temporary_file.write(
            image_bytes
        )

        return temporary_file.name


def delete_temporary_file(
    temporary_path: str | None
) -> None:
    """
    Delete the temporary uploaded screenshot after processing.
    """

    if (
        temporary_path
        and os.path.exists(
            temporary_path
        )
    ):
        os.remove(
            temporary_path
        )


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home() -> dict[str, str]:
    """
    Basic service information.
    """

    return {
        "status": "online",
        "service": "Scoreboard OCR API",
        "version": "1.0.0"
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    """
    Railway health-check endpoint.
    """

    return {
        "status": "healthy"
    }


@app.post("/parse-scoreboard")
async def parse_scoreboard_endpoint(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-Key"
    ),
    include_debug: bool = False
) -> dict[str, Any]:
    """
    Accept one screenshot and return structured scoreboard JSON.
    """

    validate_api_key(
        x_api_key
    )

    original_filename = (
        file.filename
        or "uploaded_scoreboard.jpg"
    )

    extension = validate_extension(
        original_filename
    )

    temporary_path = None

    try:
        temporary_path = await save_upload_to_temporary_file(
            uploaded_file=file,
            extension=extension
        )

        async with OCR_SEMAPHORE:
            parsed_data = await asyncio.to_thread(
                parse_scoreboard,
                temporary_path,
                include_debug
            )

        parsed_data["uploaded_filename"] = (
            original_filename
        )

        return parsed_data

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    finally:
        delete_temporary_file(
            temporary_path
        )