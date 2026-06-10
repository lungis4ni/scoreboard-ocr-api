import gc
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any, Generator

# ============================================================
# CPU LIMITS
# ============================================================
# These must be set before importing PyTorch, EasyOCR, or OpenCV.

CPU_THREAD_LIMIT = 2

os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREAD_LIMIT))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREAD_LIMIT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_THREAD_LIMIT))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(CPU_THREAD_LIMIT))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(CPU_THREAD_LIMIT))

import cv2
import easyocr
import torch


# ============================================================
# RUNTIME CONFIGURATION
# ============================================================

torch.set_num_threads(CPU_THREAD_LIMIT)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # This can only be set once per process.
    pass

cv2.setNumThreads(1)

USE_GPU = False

EXPECTED_PLAYERS_PER_TEAM = 4

# OCR occasionally reads the kill icon as a false leading 9.
# Example: a visible 5 may be returned as 95.
MAX_REASONABLE_KILLS = 30

# Do not automatically double every screenshot.
# Smaller images are enlarged moderately. Larger images are capped.
TARGET_IMAGE_WIDTH = 1800
MAX_IMAGE_WIDTH = 2000

# Avoid re-reading every clear heading.
HEADER_RETRY_CONFIDENCE_THRESHOLD = 0.90

# Avoid excessive OCR passes.
MAX_HEADER_RETRY_VARIANTS = 2
MAX_MISSING_HEADER_RETRY_VARIANTS = 3
MAX_PLAYER_RETRY_VARIANTS = 2


# ============================================================
# LOGGING
# ============================================================

def log(message: str) -> None:
    """
    Print a timestamped parser message for Railway deployment logs.
    """

    print(
        f"[scoreboard-parser] {message}",
        flush=True
    )


# ============================================================
# OCR READER
# ============================================================

@lru_cache(maxsize=1)
def get_ocr_reader() -> easyocr.Reader:
    """
    Load EasyOCR once and reuse it across API requests.
    """

    log(
        f"Loading EasyOCR reader. "
        f"CPU threads={torch.get_num_threads()}, "
        f"OpenCV threads={cv2.getNumThreads()}."
    )

    reader = easyocr.Reader(
        ["en"],
        gpu=USE_GPU
    )

    log("EasyOCR reader loaded successfully.")

    return reader


# ============================================================
# IMAGE PREPARATION
# ============================================================

def resize_image_safely(
    image: Any
) -> tuple[Any, float]:
    """
    Resize a screenshot conservatively.

    The previous parser always doubled width and height, creating
    four times as many pixels. This version enlarges only where useful
    and caps oversized screenshots.
    """

    original_height, original_width = image.shape[:2]

    if original_width <= 0:
        raise ValueError(
            "The uploaded image width is invalid."
        )

    if original_width < TARGET_IMAGE_WIDTH:
        scale_factor = (
            TARGET_IMAGE_WIDTH
            / original_width
        )
    elif original_width > MAX_IMAGE_WIDTH:
        scale_factor = (
            MAX_IMAGE_WIDTH
            / original_width
        )
    else:
        scale_factor = 1.0

    if abs(scale_factor - 1.0) < 0.01:
        return image, 1.0

    resized_image = cv2.resize(
        image,
        None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=cv2.INTER_CUBIC
    )

    return resized_image, round(
        scale_factor,
        3
    )


# ============================================================
# TEXT NORMALISATION
# ============================================================

def normalise_team_heading(
    raw_text: str
) -> str | None:
    """
    Convert common OCR mistakes in team headings.

    Examples:
        TEAMS  -> TEAM5
        TEAMA  -> TEAM4
        TEAMB  -> TEAM8
        TEAMZ  -> TEAM2
        TEAMT  -> TEAM7
        TEAMIO -> TEAM10
        TEAMZO -> TEAM20
        TEAMZ4 -> TEAM24
    """

    cleaned_text = re.sub(
        r"[^A-Z0-9]",
        "",
        raw_text.upper()
    )

    if not cleaned_text.startswith("TEAM"):
        return None

    suffix = cleaned_text[4:]

    if suffix == "":
        return None

    direct_replacements = {
        "S": "5",
        "A": "4",
        "B": "8",
        "Z": "2",
        "T": "7",
        "O": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "ID": "10",
        "IO": "10",
        "I0": "10",
        "LO": "10",
        "L0": "10",
        "ZO": "20",
        "Z0": "20",
        "I5": "15",
        "IS": "15",
        "L5": "15",
        "LS": "15"
    }

    if suffix in direct_replacements:
        suffix = direct_replacements[
            suffix
        ]

    character_replacements = {
        "O": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "S": "5",
        "B": "8",
        "Z": "2",
        "A": "4",
        "T": "7"
    }

    suffix = "".join(
        character_replacements.get(
            character,
            character
        )
        for character in suffix
    )

    if not re.fullmatch(
        r"\d{1,2}",
        suffix
    ):
        return None

    return f"TEAM{int(suffix)}"


def heading_needs_retry(
    raw_text: str,
    confidence: float
) -> bool:
    """
    Decide whether a detected team heading needs targeted OCR.

    Clear two-digit numeric headings are trusted.
    Ambiguous or low-confidence headings are re-read.
    """

    cleaned_text = re.sub(
        r"[^A-Z0-9]",
        "",
        raw_text.upper()
    )

    if not cleaned_text.startswith(
        "TEAM"
    ):
        return True

    suffix = cleaned_text[4:]

    suffix_is_numeric = bool(
        re.fullmatch(
            r"\d{1,2}",
            suffix
        )
    )

    return (
        confidence
        < HEADER_RETRY_CONFIDENCE_THRESHOLD
        or not suffix_is_numeric
        or len(suffix) == 1
    )


def parse_small_number(
    raw_text: str
) -> int | None:
    """
    Read a small OCR number.

    Examples:
        01 -> 1
        O1 -> 1
        I4 -> 14
    """

    cleaned = raw_text.strip().upper()

    if re.search(
        r"[A-HJ-KM-NP-RT-Z]",
        cleaned
    ):
        return None

    cleaned = cleaned.replace(
        "O",
        "0"
    )

    cleaned = cleaned.replace(
        "I",
        "1"
    )

    cleaned = cleaned.replace(
        "L",
        "1"
    )

    cleaned = cleaned.replace(
        "|",
        "1"
    )

    digits_only = re.sub(
        r"[^0-9]",
        "",
        cleaned
    )

    if digits_only == "":
        return None

    if len(digits_only) > 2:
        return None

    return int(
        digits_only
    )


def parse_kill_number(
    raw_text: str
) -> int | None:
    """
    Read a kill count and correct a common false leading 9.

    Examples:
        95 -> 5
        94 -> 4
        15 -> 15
    """

    number = parse_small_number(
        raw_text
    )

    if number is None:
        return None

    if (
        number > MAX_REASONABLE_KILLS
        and 90 <= number <= 99
    ):
        return number % 10

    return number


# ============================================================
# OCR BOX HELPERS
# ============================================================

def calculate_box_coordinates(
    box: list[list[float]]
) -> dict[str, int]:
    """
    Convert an EasyOCR polygon into a simple rectangle.
    """

    x_values = [
        point[0]
        for point in box
    ]

    y_values = [
        point[1]
        for point in box
    ]

    x_left = int(
        min(x_values)
    )

    x_right = int(
        max(x_values)
    )

    y_top = int(
        min(y_values)
    )

    y_bottom = int(
        max(y_values)
    )

    return {
        "x_left": x_left,
        "x_right": x_right,
        "y_top": y_top,
        "y_bottom": y_bottom,
        "x_centre": int(
            (
                x_left
                + x_right
            )
            / 2
        ),
        "y_centre": int(
            (
                y_top
                + y_bottom
            )
            / 2
        )
    }


def calculate_iou(
    first_item: dict[str, Any],
    second_item: dict[str, Any]
) -> float:
    """
    Calculate overlap between two OCR rectangles.
    """

    x_left = max(
        first_item["x_left"],
        second_item["x_left"]
    )

    y_top = max(
        first_item["y_top"],
        second_item["y_top"]
    )

    x_right = min(
        first_item["x_right"],
        second_item["x_right"]
    )

    y_bottom = min(
        first_item["y_bottom"],
        second_item["y_bottom"]
    )

    overlap_width = max(
        0,
        x_right - x_left
    )

    overlap_height = max(
        0,
        y_bottom - y_top
    )

    overlap_area = (
        overlap_width
        * overlap_height
    )

    first_area = (
        max(
            0,
            first_item["x_right"]
            - first_item["x_left"]
        )
        * max(
            0,
            first_item["y_bottom"]
            - first_item["y_top"]
        )
    )

    second_area = (
        max(
            0,
            second_item["x_right"]
            - second_item["x_left"]
        )
        * max(
            0,
            second_item["y_bottom"]
            - second_item["y_top"]
        )
    )

    combined_area = (
        first_area
        + second_area
        - overlap_area
    )

    if combined_area == 0:
        return 0.0

    return (
        overlap_area
        / combined_area
    )


def boxes_are_likely_duplicates(
    first_item: dict[str, Any],
    second_item: dict[str, Any]
) -> bool:
    """
    Identify duplicate OCR blocks produced by retry passes.
    """

    if calculate_iou(
        first_item,
        second_item
    ) >= 0.45:
        return True

    same_row = (
        abs(
            first_item["y_centre"]
            - second_item["y_centre"]
        )
        <= 12
    )

    similar_left = (
        abs(
            first_item["x_left"]
            - second_item["x_left"]
        )
        <= 18
    )

    similar_right = (
        abs(
            first_item["x_right"]
            - second_item["x_right"]
        )
        <= 18
    )

    return (
        same_row
        and similar_left
        and similar_right
    )


def merge_ocr_item(
    ocr_items: list[dict[str, Any]],
    new_item: dict[str, Any]
) -> None:
    """
    Merge one OCR result without retaining duplicate blocks.
    """

    for index, existing_item in enumerate(
        ocr_items
    ):
        if boxes_are_likely_duplicates(
            existing_item,
            new_item
        ):
            if (
                new_item["confidence"]
                > existing_item["confidence"]
            ):
                ocr_items[
                    index
                ] = new_item

            return

    ocr_items.append(
        new_item
    )


def append_raw_ocr_results(
    ocr_items: list[dict[str, Any]],
    raw_results: list[Any],
    pass_name: str,
    x_offset: int = 0,
    y_offset: int = 0
) -> None:
    """
    Convert EasyOCR output into parser-ready dictionaries.
    """

    for box, text, confidence in raw_results:
        coordinates = (
            calculate_box_coordinates(
                box
            )
        )

        coordinates["x_left"] += (
            x_offset
        )

        coordinates["x_right"] += (
            x_offset
        )

        coordinates["x_centre"] += (
            x_offset
        )

        coordinates["y_top"] += (
            y_offset
        )

        coordinates["y_bottom"] += (
            y_offset
        )

        coordinates["y_centre"] += (
            y_offset
        )

        merge_ocr_item(
            ocr_items,
            {
                "text": text.strip(),
                "confidence": round(
                    float(
                        confidence
                    ),
                    3
                ),
                "ocr_pass": pass_name,
                **coordinates
            }
        )


def run_ocr(
    reader: easyocr.Reader,
    image: Any,
    pass_name: str
) -> list[Any]:
    """
    Run EasyOCR with conservative CPU settings.
    """

    started_at = time.perf_counter()

    log(
        f"Starting OCR pass: {pass_name}"
    )

    results = reader.readtext(
        image,
        detail=1,
        paragraph=False,
        decoder="greedy",
        batch_size=1,
        workers=0
    )

    elapsed = round(
        time.perf_counter()
        - started_at,
        2
    )

    log(
        f"Finished OCR pass: {pass_name} "
        f"in {elapsed}s with "
        f"{len(results)} block(s)."
    )

    return results


# ============================================================
# RETRY IMAGE VARIANTS
# ============================================================

def generate_retry_variants(
    image_region: Any,
    maximum_variants: int
) -> Generator[
    tuple[str, Any],
    None,
    None
]:
    """
    Yield retry images one at a time.

    The previous implementation created every variant in memory at once.
    This version generates and releases them progressively.
    """

    if maximum_variants <= 0:
        return

    yield (
        "colour",
        image_region
    )

    if maximum_variants == 1:
        return

    if len(
        image_region.shape
    ) == 3:
        grayscale = cv2.cvtColor(
            image_region,
            cv2.COLOR_BGR2GRAY
        )
    else:
        grayscale = (
            image_region.copy()
        )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    contrast_enhanced = clahe.apply(
        grayscale
    )

    yield (
        "contrast",
        contrast_enhanced
    )

    if maximum_variants == 2:
        del grayscale
        del contrast_enhanced
        gc.collect()
        return

    adaptive_threshold = cv2.adaptiveThreshold(
        contrast_enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7
    )

    yield (
        "threshold",
        adaptive_threshold
    )

    del grayscale
    del contrast_enhanced
    del adaptive_threshold

    gc.collect()


# ============================================================
# ROW GROUPING
# ============================================================

def group_items_by_y(
    items: list[dict[str, Any]],
    tolerance: int
) -> list[dict[str, Any]]:
    """
    Group OCR blocks that appear on approximately the same row.
    """

    if not items:
        return []

    sorted_items = sorted(
        items,
        key=lambda item: item[
            "y_centre"
        ]
    )

    rows = []

    for item in sorted_items:
        matching_row = None

        for row in rows:
            if (
                abs(
                    item["y_centre"]
                    - row["average_y"]
                )
                <= tolerance
            ):
                matching_row = row
                break

        if matching_row is None:
            rows.append({
                "average_y": item[
                    "y_centre"
                ],
                "items": [
                    item
                ]
            })

        else:
            matching_row[
                "items"
            ].append(
                item
            )

            matching_row[
                "average_y"
            ] = int(
                sum(
                    row_item[
                        "y_centre"
                    ]
                    for row_item
                    in matching_row[
                        "items"
                    ]
                )
                / len(
                    matching_row[
                        "items"
                    ]
                )
            )

    for row in rows:
        row["items"] = sorted(
            row["items"],
            key=lambda item: item[
                "x_left"
            ]
        )

    return rows


def group_anchors_into_visual_rows(
    team_anchors: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Group team headings into horizontal scoreboard rows.
    """

    if not team_anchors:
        return []

    visual_rows = []

    for anchor in sorted(
        team_anchors,
        key=lambda item: item[
            "y_centre"
        ]
    ):
        matching_row = None

        for row in visual_rows:
            if (
                abs(
                    anchor[
                        "y_centre"
                    ]
                    - row[
                        "average_y"
                    ]
                )
                <= 75
            ):
                matching_row = row
                break

        if matching_row is None:
            visual_rows.append({
                "average_y": anchor[
                    "y_centre"
                ],
                "anchors": [
                    anchor
                ]
            })

        else:
            matching_row[
                "anchors"
            ].append(
                anchor
            )

            matching_row[
                "average_y"
            ] = int(
                sum(
                    item[
                        "y_centre"
                    ]
                    for item
                    in matching_row[
                        "anchors"
                    ]
                )
                / len(
                    matching_row[
                        "anchors"
                    ]
                )
            )

    visual_rows = sorted(
        visual_rows,
        key=lambda row: row[
            "average_y"
        ]
    )

    for row in visual_rows:
        row["anchors"] = sorted(
            row["anchors"],
            key=lambda anchor: anchor[
                "x_centre"
            ]
        )

    return visual_rows


# ============================================================
# GRID HELPERS
# ============================================================

def estimate_card_width(
    team_anchors: list[dict[str, Any]],
    image_width: int
) -> int:
    """
    Estimate card width from spacing between detected headings.
    """

    differences = []

    for row in group_anchors_into_visual_rows(
        team_anchors
    ):
        anchors = row[
            "anchors"
        ]

        for index in range(
            len(
                anchors
            )
            - 1
        ):
            difference = (
                anchors[
                    index + 1
                ][
                    "x_centre"
                ]
                - anchors[
                    index
                ][
                    "x_centre"
                ]
            )

            if difference > 100:
                differences.append(
                    difference
                )

    if differences:
        return int(
            median(
                differences
            )
        )

    return max(
        250,
        int(
            image_width
            * 0.90
        )
    )


def cluster_coordinate_values(
    values: list[int],
    tolerance: int
) -> list[int]:
    """
    Convert nearby coordinates into approximate grid positions.
    """

    clusters = []

    for value in sorted(
        values
    ):
        matching_cluster = None

        for cluster in clusters:
            if (
                abs(
                    value
                    - cluster[
                        "average"
                    ]
                )
                <= tolerance
            ):
                matching_cluster = (
                    cluster
                )

                break

        if matching_cluster is None:
            clusters.append({
                "average": value,
                "values": [
                    value
                ]
            })

        else:
            matching_cluster[
                "values"
            ].append(
                value
            )

            matching_cluster[
                "average"
            ] = int(
                sum(
                    matching_cluster[
                        "values"
                    ]
                )
                / len(
                    matching_cluster[
                        "values"
                    ]
                )
            )

    return sorted(
        cluster[
            "average"
        ]
        for cluster
        in clusters
    )


def find_anchor_near_position(
    team_anchors: list[dict[str, Any]],
    expected_x: int,
    expected_y: int,
    x_tolerance: int,
    y_tolerance: int
) -> dict[str, Any] | None:
    """
    Check whether a heading already occupies a grid position.
    """

    for anchor in team_anchors:
        if (
            abs(
                anchor[
                    "x_centre"
                ]
                - expected_x
            )
            <= x_tolerance
            and abs(
                anchor[
                    "y_centre"
                ]
                - expected_y
            )
            <= y_tolerance
        ):
            return anchor

    return None


# ============================================================
# RANK DETECTION
# ============================================================

def find_nearby_rank_for_position(
    expected_x: int,
    expected_y: int,
    ocr_items: list[dict[str, Any]],
    card_width: int
) -> int | None:
    """
    Find a placement number to the left of a team heading.
    """

    candidates = []

    for item in ocr_items:
        number = parse_small_number(
            item["text"]
        )

        if (
            number is None
            or not 1 <= number <= 99
        ):
            continue

        horizontal_distance = (
            expected_x
            - item[
                "x_centre"
            ]
        )

        vertical_distance = abs(
            expected_y
            - item[
                "y_centre"
            ]
        )

        if (
            card_width * 0.30
            <= horizontal_distance
            <= card_width * 0.95
            and vertical_distance <= 60
        ):
            candidates.append({
                "rank": number,
                "distance": (
                    horizontal_distance
                    + vertical_distance
                )
            })

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda item: item[
            "distance"
        ]
    )[
        "rank"
    ]


def infer_missing_ranks(
    ordered_anchors: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Infer missing placements from surrounding visible ranks.
    """

    known_offsets = []

    for index, anchor in enumerate(
        ordered_anchors
    ):
        if anchor[
            "placement"
        ] is not None:
            known_offsets.append(
                anchor[
                    "placement"
                ]
                - index
            )

    if not known_offsets:
        return ordered_anchors

    likely_offset = int(
        round(
            median(
                known_offsets
            )
        )
    )

    for index, anchor in enumerate(
        ordered_anchors
    ):
        if anchor[
            "placement"
        ] is None:
            inferred_rank = (
                index
                + likely_offset
            )

            if inferred_rank > 0:
                anchor[
                    "placement"
                ] = inferred_rank

                anchor[
                    "placement_inferred"
                ] = True

    return ordered_anchors


# ============================================================
# TEAM HEADER OCR
# ============================================================

def collect_team_heading_candidates(
    reader: easyocr.Reader,
    source_image: Any,
    expected_x: int,
    expected_y: int,
    card_width: int,
    maximum_variants: int
) -> list[dict[str, Any]]:
    """
    Re-read a small heading region progressively.
    """

    image_height, image_width = (
        source_image.shape[:2]
    )

    x_left = max(
        0,
        int(
            expected_x
            - card_width * 0.38
        )
    )

    x_right = min(
        image_width,
        int(
            expected_x
            + card_width * 0.18
        )
    )

    y_top = max(
        0,
        int(
            expected_y
            - card_width * 0.11
        )
    )

    y_bottom = min(
        image_height,
        int(
            expected_y
            + card_width * 0.11
        )
    )

    heading_region = source_image[
        y_top:y_bottom,
        x_left:x_right
    ]

    if heading_region.size == 0:
        return []

    candidates = []

    for variant_name, variant in (
        generate_retry_variants(
            heading_region,
            maximum_variants
        )
    ):
        retry_results = run_ocr(
            reader,
            variant,
            (
                "heading-"
                + variant_name
            )
        )

        for box, text, confidence in (
            retry_results
        ):
            normalised_team = (
                normalise_team_heading(
                    text
                )
            )

            if normalised_team is None:
                continue

            coordinates = (
                calculate_box_coordinates(
                    box
                )
            )

            absolute_x = (
                coordinates[
                    "x_centre"
                ]
                + x_left
            )

            absolute_y = (
                coordinates[
                    "y_centre"
                ]
                + y_top
            )

            if (
                abs(
                    absolute_x
                    - expected_x
                )
                > card_width * 0.32
                or abs(
                    absolute_y
                    - expected_y
                )
                > card_width * 0.16
            ):
                continue

            candidates.append({
                "team": normalised_team,
                "original_team_ocr_text": (
                    text
                ),
                "team_heading_confidence": round(
                    float(
                        confidence
                    ),
                    3
                ),
                "x_centre": absolute_x,
                "y_centre": absolute_y,
                "ocr_pass": (
                    variant_name
                )
            })

        del retry_results
        gc.collect()

    return candidates


def choose_best_heading_candidate(
    candidates: list[dict[str, Any]],
    fallback: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """
    Choose the best heading using confidence and repeated readings.
    """

    all_candidates = list(
        candidates
    )

    if fallback is not None:
        all_candidates.append(
            fallback
        )

    if not all_candidates:
        return None

    grouped_candidates = {}

    for candidate in all_candidates:
        grouped_candidates.setdefault(
            candidate[
                "team"
            ],
            []
        ).append(
            candidate
        )

    scored_results = []

    for team_name, group in (
        grouped_candidates.items()
    ):
        best_candidate = max(
            group,
            key=lambda item: item[
                "team_heading_confidence"
            ]
        )

        score = sum(
            item[
                "team_heading_confidence"
            ]
            for item in group
        )

        score += (
            0.15
            * len(
                group
            )
        )

        score += (
            0.03
            * len(
                re.sub(
                    r"\D",
                    "",
                    team_name
                )
            )
        )

        scored_results.append(
            (
                score,
                best_candidate
            )
        )

    return max(
        scored_results,
        key=lambda item: item[0]
    )[1]


def remove_duplicate_team_anchors(
    team_anchors: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Remove headings detected more than once at the same position.
    """

    unique_anchors = []

    for anchor in sorted(
        team_anchors,
        key=lambda item: (
            item[
                "y_centre"
            ],
            item[
                "x_centre"
            ]
        )
    ):
        matching_index = None

        for index, existing in enumerate(
            unique_anchors
        ):
            if (
                abs(
                    existing[
                        "x_centre"
                    ]
                    - anchor[
                        "x_centre"
                    ]
                )
                < 35
                and abs(
                    existing[
                        "y_centre"
                    ]
                    - anchor[
                        "y_centre"
                    ]
                )
                < 35
            ):
                matching_index = index
                break

        if matching_index is None:
            unique_anchors.append(
                anchor
            )

        else:
            existing = unique_anchors[
                matching_index
            ]

            if (
                anchor.get(
                    "team_heading_confidence",
                    0.0
                )
                > existing.get(
                    "team_heading_confidence",
                    0.0
                )
            ):
                unique_anchors[
                    matching_index
                ] = anchor

    return unique_anchors


def refine_uncertain_team_headers(
    team_anchors: list[dict[str, Any]],
    reader: easyocr.Reader,
    source_image: Any,
    image_width: int
) -> list[dict[str, Any]]:
    """
    Retry only uncertain team headings.

    Clear headings are kept without extra OCR calls.
    """

    card_width = estimate_card_width(
        team_anchors,
        image_width
    )

    refined_anchors = []

    for anchor in team_anchors:
        if not heading_needs_retry(
            anchor[
                "original_team_ocr_text"
            ],
            anchor[
                "team_heading_confidence"
            ]
        ):
            refined_anchors.append(
                anchor
            )

            continue

        log(
            f"Refining uncertain heading "
            f"'{anchor['original_team_ocr_text']}'."
        )

        candidates = (
            collect_team_heading_candidates(
                reader=reader,
                source_image=source_image,
                expected_x=anchor[
                    "x_centre"
                ],
                expected_y=anchor[
                    "y_centre"
                ],
                card_width=card_width,
                maximum_variants=(
                    MAX_HEADER_RETRY_VARIANTS
                )
            )
        )

        best_candidate = (
            choose_best_heading_candidate(
                candidates,
                fallback=anchor
            )
        )

        if best_candidate is None:
            refined_anchors.append(
                anchor
            )

            continue

        best_candidate[
            "header_refined"
        ] = (
            best_candidate[
                "team"
            ]
            != anchor[
                "team"
            ]
        )

        refined_anchors.append(
            best_candidate
        )

    return refined_anchors


def recover_missing_team_headers(
    team_anchors: list[dict[str, Any]],
    reader: easyocr.Reader,
    source_image: Any,
    ocr_items: list[dict[str, Any]],
    image_width: int
) -> list[dict[str, Any]]:
    """
    Re-read only empty grid positions.

    This handles cases such as a missed TEAM24 card at placement 8.
    """

    if len(
        team_anchors
    ) < 4:
        return team_anchors

    card_width = estimate_card_width(
        team_anchors,
        image_width
    )

    x_positions = cluster_coordinate_values(
        [
            anchor[
                "x_centre"
            ]
            for anchor
            in team_anchors
        ],
        tolerance=max(
            70,
            int(
                card_width
                * 0.25
            )
        )
    )

    y_positions = cluster_coordinate_values(
        [
            anchor[
                "y_centre"
            ]
            for anchor
            in team_anchors
        ],
        tolerance=max(
            60,
            int(
                card_width
                * 0.20
            )
        )
    )

    if (
        len(
            x_positions
        )
        < 2
        or len(
            y_positions
        )
        < 2
    ):
        return team_anchors

    recovered_anchors = list(
        team_anchors
    )

    for expected_y in y_positions:
        for expected_x in x_positions:
            existing_anchor = (
                find_anchor_near_position(
                    team_anchors=(
                        recovered_anchors
                    ),
                    expected_x=expected_x,
                    expected_y=expected_y,
                    x_tolerance=int(
                        card_width
                        * 0.25
                    ),
                    y_tolerance=int(
                        card_width
                        * 0.16
                    )
                )
            )

            if existing_anchor is not None:
                continue

            log(
                "Scanning a missing grid position."
            )

            candidates = (
                collect_team_heading_candidates(
                    reader=reader,
                    source_image=source_image,
                    expected_x=expected_x,
                    expected_y=expected_y,
                    card_width=card_width,
                    maximum_variants=(
                        MAX_MISSING_HEADER_RETRY_VARIANTS
                    )
                )
            )

            best_candidate = (
                choose_best_heading_candidate(
                    candidates
                )
            )

            if best_candidate is not None:
                best_candidate[
                    "recovered_from_missing_grid_position"
                ] = True

                best_candidate[
                    "header_refined"
                ] = False

                recovered_anchors.append(
                    best_candidate
                )

                log(
                    f"Recovered missing heading "
                    f"{best_candidate['team']}."
                )

                continue

            nearby_rank = (
                find_nearby_rank_for_position(
                    expected_x=expected_x,
                    expected_y=expected_y,
                    ocr_items=ocr_items,
                    card_width=card_width
                )
            )

            if nearby_rank is not None:
                recovered_anchors.append({
                    "team": (
                        "UNKNOWN_TEAM_AT_PLACE_"
                        + str(
                            nearby_rank
                        )
                    ),
                    "original_team_ocr_text": "",
                    "team_heading_confidence": 0.0,
                    "x_centre": expected_x,
                    "y_centre": expected_y,
                    "recovered_from_missing_grid_position": True,
                    "header_refined": False,
                    "team_name_needs_review": True
                })

                log(
                    f"Recovered an unreadable card "
                    f"at placement {nearby_rank}."
                )

    return recovered_anchors


# ============================================================
# CARD REGIONS
# ============================================================

def build_team_regions(
    team_anchors: list[dict[str, Any]],
    ocr_items: list[dict[str, Any]],
    image_width: int,
    image_height: int
) -> tuple[
    list[dict[str, Any]],
    int
]:
    """
    Create dynamic card regions from visible headings.
    """

    visual_rows = (
        group_anchors_into_visual_rows(
            team_anchors
        )
    )

    card_width = estimate_card_width(
        team_anchors,
        image_width
    )

    ordered_anchors = []

    for row_index, row in enumerate(
        visual_rows
    ):
        for column_index, anchor in (
            enumerate(
                row[
                    "anchors"
                ]
            )
        ):
            anchor[
                "visual_row"
            ] = row_index

            anchor[
                "visual_column"
            ] = column_index

            ordered_anchors.append(
                anchor
            )

    for anchor in ordered_anchors:
        anchor[
            "placement"
        ] = find_nearby_rank_for_position(
            expected_x=anchor[
                "x_centre"
            ],
            expected_y=anchor[
                "y_centre"
            ],
            ocr_items=ocr_items,
            card_width=card_width
        )

        anchor[
            "placement_inferred"
        ] = False

    ordered_anchors = infer_missing_ranks(
        ordered_anchors
    )

    row_header_positions = [
        row[
            "average_y"
        ]
        for row
        in visual_rows
    ]

    for anchor in ordered_anchors:
        row_index = anchor[
            "visual_row"
        ]

        if (
            row_index + 1
            < len(
                row_header_positions
            )
        ):
            y_bottom = int(
                row_header_positions[
                    row_index + 1
                ]
                - 30
            )

        else:
            y_bottom = image_height

        anchor[
            "region"
        ] = {
            "x_left": max(
                0,
                int(
                    anchor[
                        "x_centre"
                    ]
                    - card_width
                    * 0.92
                )
            ),
            "x_right": min(
                image_width,
                int(
                    anchor[
                        "x_centre"
                    ]
                    + card_width
                    * 0.12
                )
            ),
            "y_top": max(
                0,
                int(
                    anchor[
                        "y_centre"
                    ]
                    + 20
                )
            ),
            "y_bottom": min(
                image_height,
                y_bottom
            )
        }

    return (
        ordered_anchors,
        card_width
    )


# ============================================================
# PLAYER EXTRACTION
# ============================================================

def weighted_non_overlapping_name_items(
    items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Keep split names while avoiding duplicate retry readings.
    """

    if not items:
        return []

    sorted_items = sorted(
        items,
        key=lambda item: (
            item[
                "x_right"
            ],
            item[
                "x_left"
            ]
        )
    )

    previous_indexes = []

    for index, item in enumerate(
        sorted_items
    ):
        compatible_index = -1

        for earlier_index in range(
            index - 1,
            -1,
            -1
        ):
            if (
                sorted_items[
                    earlier_index
                ][
                    "x_right"
                ]
                < item[
                    "x_left"
                ]
                - 3
            ):
                compatible_index = (
                    earlier_index
                )

                break

        previous_indexes.append(
            compatible_index
        )

    best_scores = [
        0.0
    ] * (
        len(
            sorted_items
        )
        + 1
    )

    selections = [
        []
        for _ in range(
            len(
                sorted_items
            )
            + 1
        )
    ]

    for index, item in enumerate(
        sorted_items,
        start=1
    ):
        width = max(
            1,
            item[
                "x_right"
            ]
            - item[
                "x_left"
            ]
        )

        item_score = (
            width
            * max(
                0.15,
                item[
                    "confidence"
                ]
            )
        )

        previous_index = (
            previous_indexes[
                index - 1
            ]
            + 1
        )

        include_score = (
            item_score
            + best_scores[
                previous_index
            ]
        )

        exclude_score = (
            best_scores[
                index - 1
            ]
        )

        if include_score > exclude_score:
            best_scores[
                index
            ] = include_score

            selections[
                index
            ] = (
                selections[
                    previous_index
                ]
                + [
                    item
                ]
            )

        else:
            best_scores[
                index
            ] = exclude_score

            selections[
                index
            ] = selections[
                index - 1
            ]

    return sorted(
        selections[
            -1
        ],
        key=lambda item: item[
            "x_left"
        ]
    )


def extract_players_for_team(
    anchor: dict[str, Any],
    ocr_items: list[dict[str, Any]],
    card_width: int
) -> list[dict[str, Any]]:
    """
    Extract player names and kill counts for one team card.
    """

    region = anchor[
        "region"
    ]

    region_items = [
        item
        for item in ocr_items
        if (
            region[
                "x_left"
            ]
            <= item[
                "x_centre"
            ]
            <= region[
                "x_right"
            ]
            and region[
                "y_top"
            ]
            <= item[
                "y_centre"
            ]
            <= region[
                "y_bottom"
            ]
        )
    ]

    grouped_rows = group_items_by_y(
        items=region_items,
        tolerance=max(
            14,
            int(
                card_width
                * 0.04
            )
        )
    )

    players = []

    for row in grouped_rows:
        numeric_candidates = []

        for item in row[
            "items"
        ]:
            number = parse_kill_number(
                item[
                    "text"
                ]
            )

            if number is None:
                continue

            distance = abs(
                item[
                    "x_centre"
                ]
                - anchor[
                    "x_centre"
                ]
            )

            if (
                distance
                <= card_width
                * 0.22
            ):
                numeric_candidates.append({
                    "item": item,
                    "number": number,
                    "distance": distance
                })

        if not numeric_candidates:
            continue

        kill_match = min(
            numeric_candidates,
            key=lambda item: (
                item[
                    "distance"
                ],
                -item[
                    "item"
                ][
                    "confidence"
                ]
            )
        )

        kill_item = kill_match[
            "item"
        ]

        possible_name_items = [
            item
            for item
            in row[
                "items"
            ]
            if (
                item[
                    "x_right"
                ]
                < kill_item[
                    "x_left"
                ]
                - 4
                and parse_small_number(
                    item[
                        "text"
                    ]
                )
                is None
                and normalise_team_heading(
                    item[
                        "text"
                    ]
                )
                is None
            )
        ]

        name_items = (
            weighted_non_overlapping_name_items(
                possible_name_items
            )
        )

        if not name_items:
            continue

        player_name = " ".join(
            item[
                "text"
            ].strip()
            for item
            in name_items
            if item[
                "text"
            ].strip()
        ).strip()

        if player_name == "":
            continue

        name_confidence = round(
            sum(
                item[
                    "confidence"
                ]
                for item
                in name_items
            )
            / len(
                name_items
            ),
            3
        )

        players.append({
            "ign": player_name,
            "kills": kill_match[
                "number"
            ],
            "name_confidence": (
                name_confidence
            ),
            "kill_confidence": (
                kill_item[
                    "confidence"
                ]
            ),
            "needs_review": (
                name_confidence
                < 0.70
            )
        })

    return players


def retry_missing_player_rows(
    anchor: dict[str, Any],
    reader: easyocr.Reader,
    source_image: Any,
    ocr_items: list[dict[str, Any]],
    card_width: int
) -> list[dict[str, Any]]:
    """
    Retry a team card progressively until enough rows are recovered.
    """

    region = anchor[
        "region"
    ]

    x_left = region[
        "x_left"
    ]

    x_right = region[
        "x_right"
    ]

    y_top = region[
        "y_top"
    ]

    y_bottom = region[
        "y_bottom"
    ]

    image_region = source_image[
        y_top:y_bottom,
        x_left:x_right
    ]

    if image_region.size == 0:
        return []

    players = extract_players_for_team(
        anchor,
        ocr_items,
        card_width
    )

    for variant_name, variant in (
        generate_retry_variants(
            image_region,
            MAX_PLAYER_RETRY_VARIANTS
        )
    ):
        if (
            len(
                players
            )
            >= EXPECTED_PLAYERS_PER_TEAM
        ):
            break

        log(
            f"Retrying player rows for "
            f"{anchor['team']} "
            f"using {variant_name}."
        )

        retry_results = run_ocr(
            reader,
            variant,
            (
                "player-rows-"
                + variant_name
            )
        )

        append_raw_ocr_results(
            ocr_items=ocr_items,
            raw_results=retry_results,
            pass_name=(
                "player-rows-"
                + variant_name
            ),
            x_offset=x_left,
            y_offset=y_top
        )

        players = extract_players_for_team(
            anchor,
            ocr_items,
            card_width
        )

        del retry_results
        gc.collect()

    return players


# ============================================================
# MAIN PARSER FUNCTION
# ============================================================

def parse_scoreboard(
    image_path_value: str,
    include_debug: bool = False
) -> dict[str, Any]:
    """
    Process one scoreboard screenshot and return structured JSON.
    """

    parser_started_at = (
        time.perf_counter()
    )

    image_path = Path(
        image_path_value
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Could not find "
            f"'{image_path_value}'."
        )

    image = cv2.imread(
        str(
            image_path
        )
    )

    if image is None:
        raise ValueError(
            "OpenCV could not read "
            "the uploaded image."
        )

    original_height, original_width = (
        image.shape[:2]
    )

    image, scale_factor = (
        resize_image_safely(
            image
        )
    )

    image_height, image_width = (
        image.shape[:2]
    )

    log(
        f"Processing image "
        f"{original_width}x{original_height}; "
        f"working size "
        f"{image_width}x{image_height}; "
        f"scale factor={scale_factor}."
    )

    reader = get_ocr_reader()

    raw_results = run_ocr(
        reader,
        image,
        "full-image"
    )

    ocr_items = []

    append_raw_ocr_results(
        ocr_items=ocr_items,
        raw_results=raw_results,
        pass_name="full-image"
    )

    del raw_results

    gc.collect()

    team_anchors = []

    for item in ocr_items:
        normalised_team = (
            normalise_team_heading(
                item[
                    "text"
                ]
            )
        )

        if normalised_team is None:
            continue

        team_anchors.append({
            "team": normalised_team,
            "original_team_ocr_text": (
                item[
                    "text"
                ]
            ),
            "team_heading_confidence": (
                item[
                    "confidence"
                ]
            ),
            "x_centre": item[
                "x_centre"
            ],
            "y_centre": item[
                "y_centre"
            ],
            "header_refined": False
        })

    team_anchors = (
        remove_duplicate_team_anchors(
            team_anchors
        )
    )

    if not team_anchors:
        raise ValueError(
            "No team headings were detected. "
            "Try a clearer screenshot."
        )

    log(
        f"Detected {len(team_anchors)} "
        f"heading(s) during the "
        f"full-image pass."
    )

    team_anchors = (
        refine_uncertain_team_headers(
            team_anchors=team_anchors,
            reader=reader,
            source_image=image,
            image_width=image_width
        )
    )

    team_anchors = (
        remove_duplicate_team_anchors(
            team_anchors
        )
    )

    team_anchors = (
        recover_missing_team_headers(
            team_anchors=team_anchors,
            reader=reader,
            source_image=image,
            ocr_items=ocr_items,
            image_width=image_width
        )
    )

    team_anchors = (
        remove_duplicate_team_anchors(
            team_anchors
        )
    )

    log(
        f"Using {len(team_anchors)} "
        f"heading(s) after recovery."
    )

    team_regions, estimated_card_width = (
        build_team_regions(
            team_anchors=team_anchors,
            ocr_items=ocr_items,
            image_width=image_width,
            image_height=image_height
        )
    )

    structured_teams = []

    for anchor in team_regions:
        players = extract_players_for_team(
            anchor=anchor,
            ocr_items=ocr_items,
            card_width=estimated_card_width
        )

        initial_player_count = len(
            players
        )

        dynamic_retry_used = False

        if (
            initial_player_count
            < EXPECTED_PLAYERS_PER_TEAM
        ):
            dynamic_retry_used = True

            players = (
                retry_missing_player_rows(
                    anchor=anchor,
                    reader=reader,
                    source_image=image,
                    ocr_items=ocr_items,
                    card_width=(
                        estimated_card_width
                    )
                )
            )

        structured_teams.append({
            "placement": anchor[
                "placement"
            ],
            "placement_inferred": anchor[
                "placement_inferred"
            ],
            "team": anchor[
                "team"
            ],
            "original_team_ocr_text": (
                anchor.get(
                    "original_team_ocr_text",
                    ""
                )
            ),
            "team_heading_confidence": (
                anchor.get(
                    "team_heading_confidence",
                    0.0
                )
            ),
            "team_name_needs_review": (
                anchor.get(
                    "team_name_needs_review",
                    False
                )
            ),
            "header_refined": (
                anchor.get(
                    "header_refined",
                    False
                )
            ),
            "recovered_from_missing_grid_position": (
                anchor.get(
                    "recovered_from_missing_grid_position",
                    False
                )
            ),
            "initial_player_count": (
                initial_player_count
            ),
            "detected_player_count": len(
                players
            ),
            "expected_player_count": (
                EXPECTED_PLAYERS_PER_TEAM
            ),
            "dynamic_retry_used": (
                dynamic_retry_used
            ),
            "row_count_warning": (
                len(
                    players
                )
                < EXPECTED_PLAYERS_PER_TEAM
            ),
            "players": players
        })

    structured_teams = sorted(
        structured_teams,
        key=lambda team: (
            team[
                "placement"
            ]
            is None,
            team[
                "placement"
            ]
            if team[
                "placement"
            ]
            is not None
            else 999
        )
    )

    elapsed_seconds = round(
        time.perf_counter()
        - parser_started_at,
        2
    )

    output = {
        "source_image": (
            image_path.name
        ),
        "original_image_width": (
            original_width
        ),
        "original_image_height": (
            original_height
        ),
        "working_image_width": (
            image_width
        ),
        "working_image_height": (
            image_height
        ),
        "scale_factor": (
            scale_factor
        ),
        "estimated_card_width": (
            estimated_card_width
        ),
        "detected_team_count": len(
            structured_teams
        ),
        "processing_seconds": (
            elapsed_seconds
        ),
        "cpu_thread_limit": (
            CPU_THREAD_LIMIT
        ),
        "teams": structured_teams
    }

    if include_debug:
        output[
            "debug"
        ] = {
            "detected_team_regions": (
                team_regions
            ),
            "all_ocr_items": (
                ocr_items
            )
        }

    log(
        f"Completed scoreboard parsing "
        f"in {elapsed_seconds}s."
    )

    del image

    gc.collect()

    return output