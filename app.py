import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import (
    FastAPI,
    File,
    Header,
    HTTPException,
    UploadFile
)

from fastapi.middleware.cors import (
    CORSMiddleware
)

from scoreboard_parser import (
    CPU_THREAD_LIMIT,
    parse_scoreboard
)


# ============================================================
# SETTINGS
# ============================================================

MAX_UPLOAD_SIZE_MB = 10

MAX_UPLOAD_SIZE_BYTES = (
    MAX_UPLOAD_SIZE_MB
    * 1024
    * 1024
)

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}

EXPECTED_API_KEY = os.getenv(
    "API_KEY",
    ""
).strip()

# Process only one OCR screenshot at a time.
OCR_SEMAPHORE = asyncio.Semaphore(
    1
)

# Railway supports long HTTP requests, but this prevents a broken OCR
# request from running forever.
OCR_TIMEOUT_SECONDS = 840


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Scoreboard OCR API",
    description=(
        "Extracts structured placement, team, "
        "player IGN, kill-count, confidence, "
        "and review-warning data from "
        "scoreboard screenshots."
    ),
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],
    allow_credentials=False,
    allow_methods=[
        "GET",
        "POST"
    ],
    allow_headers=[
        "*"
    ]
)


# ============================================================
# HELPERS
# ============================================================

def validate_api_key(
    supplied_api_key: str | None
) -> None:
    """
    Require an API key only after API_KEY has been configured.
    """

    if not EXPECTED_API_KEY:
        return

    if (
        supplied_api_key
        != EXPECTED_API_KEY
    ):
        raise HTTPException(
            status_code=401,
            detail=(
                "Invalid or missing API key."
            )
        )


def validate_extension(
    filename: str
) -> str:
    """
    Check whether the uploaded file type is supported.
    """

    extension = Path(
        filename
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image type. "
                "Upload a JPG, JPEG, PNG, "
                "or WEBP image."
            )
        )

    return extension


async def save_upload_to_temporary_file(
    uploaded_file: UploadFile,
    extension: str
) -> str:
    """
    Store one uploaded screenshot temporarily.
    """

    image_bytes = await uploaded_file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded image is empty."
            )
        )

    if (
        len(
            image_bytes
        )
        > MAX_UPLOAD_SIZE_BYTES
    ):
        raise HTTPException(
            status_code=413,
            detail=(
                f"The uploaded image exceeds "
                f"the {MAX_UPLOAD_SIZE_MB} MB "
                f"limit."
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
    Remove a screenshot after processing.
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
def home() -> dict[str, Any]:
    """
    Basic endpoint information.
    """

    return {
        "status": "online",
        "service": "Scoreboard OCR API",
        "version": "1.1.0",
        "cpu_thread_limit": (
            CPU_THREAD_LIMIT
        )
    }


@app.get("/health")
def health_check() -> dict[str, Any]:
    """
    Railway health-check endpoint.
    """

    return {
        "status": "healthy",
        "service": "Scoreboard OCR API"
    }


@app.get("/runtime-config")
def runtime_config() -> dict[str, Any]:
    """
    Confirm the hosted CPU controls.
    """

    return {
        "cpu_thread_limit": (
            CPU_THREAD_LIMIT
        ),
        "ocr_concurrency": 1,
        "ocr_timeout_seconds": (
            OCR_TIMEOUT_SECONDS
        ),
        "api_key_enabled": bool(
            EXPECTED_API_KEY
        )
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

    request_started_at = (
        time.perf_counter()
    )

    try:
        temporary_path = (
            await save_upload_to_temporary_file(
                uploaded_file=file,
                extension=extension
            )
        )

        async with OCR_SEMAPHORE:
            parsed_data = (
                await asyncio.wait_for(
                    asyncio.to_thread(
                        parse_scoreboard,
                        temporary_path,
                        include_debug
                    ),
                    timeout=(
                        OCR_TIMEOUT_SECONDS
                    )
                )
            )

        parsed_data[
            "uploaded_filename"
        ] = original_filename

        parsed_data[
            "api_request_seconds"
        ] = round(
            time.perf_counter()
            - request_started_at,
            2
        )

        return parsed_data

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=(
                "OCR processing exceeded the "
                "maximum allowed duration."
            )
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(
                error
            )
        )

    finally:
        delete_temporary_file(
            temporary_path
        )