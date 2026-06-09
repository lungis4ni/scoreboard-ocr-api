import csv
import json
import re
from pathlib import Path
from statistics import median

import cv2
import easyocr


# ============================================================
# SETTINGS
# ============================================================

IMAGE_PATH = "scoreboard2.jpg"

OUTPUT_JSON_PATH = "scoreboard_output.json"
OUTPUT_CSV_PATH = "scoreboard_output.csv"
DEBUG_JSON_PATH = "ocr_debug.json"

# Increase this if screenshots contain very small text.
UPSCALE_FACTOR = 2

# Keep this False unless your PC has a compatible GPU setup.
USE_GPU = False

# Most squads contain four players.
EXPECTED_PLAYERS_PER_TEAM = 4

# Values above this may indicate that the kill icon was read as a
# leading digit. For example, the visible value 5 may be read as 95.
MAX_REASONABLE_KILLS = 30


# ============================================================
# TEXT NORMALISATION
# ============================================================

def normalise_team_heading(raw_text):
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
    """

    cleaned_text = re.sub(
        r"[^A-Z0-9]",
        "",
        raw_text.upper()
    )

    if not cleaned_text.startswith("TEAM"):
        return None

    suffix = cleaned_text[4:]

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
        suffix = direct_replacements[suffix]

    suffix = suffix.replace("O", "0")
    suffix = suffix.replace("D", "0")
    suffix = suffix.replace("I", "1")
    suffix = suffix.replace("L", "1")
    suffix = suffix.replace("S", "5")
    suffix = suffix.replace("B", "8")
    suffix = suffix.replace("Z", "2")

    if not re.fullmatch(r"\d{1,2}", suffix):
        return None

    return f"TEAM{int(suffix)}"


def parse_small_number(raw_text):
    """
    Read a small OCR number.

    Examples:
        01 -> 1
        O1 -> 1
        I4 -> 14
    """

    cleaned = raw_text.strip().upper()

    # Reject text containing normal letters that are unlikely to be
    # mistaken for digits.
    if re.search(r"[A-HJ-KM-NP-RT-Z]", cleaned):
        return None

    cleaned = cleaned.replace("O", "0")
    cleaned = cleaned.replace("I", "1")
    cleaned = cleaned.replace("L", "1")
    cleaned = cleaned.replace("|", "1")

    digits_only = re.sub(
        r"[^0-9]",
        "",
        cleaned
    )

    if digits_only == "":
        return None

    if len(digits_only) > 2:
        return None

    return int(digits_only)


def parse_kill_number(raw_text):
    """
    Read kill values and remove a common false leading 9 caused by
    the kill icon being interpreted as part of the value.

    Example:
        95 -> 5
        15 -> 15
    """

    number = parse_small_number(raw_text)

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

def calculate_box_coordinates(box):
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

    x_left = int(min(x_values))
    x_right = int(max(x_values))
    y_top = int(min(y_values))
    y_bottom = int(max(y_values))

    return {
        "x_left": x_left,
        "x_right": x_right,
        "y_top": y_top,
        "y_bottom": y_bottom,
        "x_centre": int((x_left + x_right) / 2),
        "y_centre": int((y_top + y_bottom) / 2)
    }


def calculate_iou(first_item, second_item):
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
        return 0

    return overlap_area / combined_area


def boxes_are_likely_duplicates(
    first_item,
    second_item
):
    """
    Retry OCR passes sometimes return slightly different rectangles
    for the same visible text. This reduces duplicate results.
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
    ocr_items,
    new_item
):
    """
    Add an OCR result unless the same area has already been detected.

    Where duplicate detections exist, keep the higher-confidence result.
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
                ocr_items[index] = new_item

            return

    ocr_items.append(
        new_item
    )


def append_raw_ocr_results(
    ocr_items,
    raw_results,
    pass_name,
    x_offset=0,
    y_offset=0
):
    """
    Convert EasyOCR results into parser-ready objects.

    Offsets are applied where OCR was run against a dynamically
    selected image region.
    """

    for box, text, confidence in raw_results:
        coordinates = calculate_box_coordinates(
            box
        )

        coordinates["x_left"] += x_offset
        coordinates["x_right"] += x_offset
        coordinates["x_centre"] += x_offset

        coordinates["y_top"] += y_offset
        coordinates["y_bottom"] += y_offset
        coordinates["y_centre"] += y_offset

        new_item = {
            "text": text.strip(),
            "confidence": round(
                float(confidence),
                3
            ),
            "ocr_pass": pass_name,
            **coordinates
        }

        merge_ocr_item(
            ocr_items,
            new_item
        )


# ============================================================
# OCR RETRY IMAGE VARIANTS
# ============================================================

def create_retry_variants(
    image_region
):
    """
    Create alternate image versions to improve OCR recovery.
    """

    if len(image_region.shape) == 3:
        grayscale = cv2.cvtColor(
            image_region,
            cv2.COLOR_BGR2GRAY
        )
    else:
        grayscale = image_region

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    contrast_enhanced = clahe.apply(
        grayscale
    )

    adaptive_threshold = cv2.adaptiveThreshold(
        contrast_enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7
    )

    inverted_threshold = cv2.bitwise_not(
        adaptive_threshold
    )

    return [
        (
            "retry_colour",
            image_region
        ),
        (
            "retry_contrast",
            contrast_enhanced
        ),
        (
            "retry_threshold",
            adaptive_threshold
        ),
        (
            "retry_inverted",
            inverted_threshold
        )
    ]


# ============================================================
# ROW GROUPING
# ============================================================

def group_items_by_y(
    items,
    tolerance
):
    """
    Group OCR text blocks that appear on approximately the same row.
    """

    if not items:
        return []

    sorted_items = sorted(
        items,
        key=lambda item: item["y_centre"]
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
                "average_y": item["y_centre"],
                "items": [item]
            })

        else:
            matching_row["items"].append(
                item
            )

            matching_row["average_y"] = int(
                sum(
                    row_item["y_centre"]
                    for row_item
                    in matching_row["items"]
                )
                / len(
                    matching_row["items"]
                )
            )

    for row in rows:
        row["items"] = sorted(
            row["items"],
            key=lambda item: item["x_left"]
        )

    return rows


def group_anchors_into_visual_rows(
    team_anchors
):
    """
    Group detected team headings into their visible horizontal rows.
    """

    if not team_anchors:
        return []

    sorted_anchors = sorted(
        team_anchors,
        key=lambda anchor: anchor["y_centre"]
    )

    visual_rows = []

    for anchor in sorted_anchors:
        matching_row = None

        for row in visual_rows:
            if (
                abs(
                    anchor["y_centre"]
                    - row["average_y"]
                )
                <= 75
            ):
                matching_row = row
                break

        if matching_row is None:
            visual_rows.append({
                "average_y": anchor["y_centre"],
                "anchors": [anchor]
            })

        else:
            matching_row["anchors"].append(
                anchor
            )

            matching_row["average_y"] = int(
                sum(
                    item["y_centre"]
                    for item
                    in matching_row["anchors"]
                )
                / len(
                    matching_row["anchors"]
                )
            )

    visual_rows = sorted(
        visual_rows,
        key=lambda row: row["average_y"]
    )

    for row in visual_rows:
        row["anchors"] = sorted(
            row["anchors"],
            key=lambda anchor: anchor["x_centre"]
        )

    return visual_rows


# ============================================================
# GRID POSITION HELPERS
# ============================================================

def estimate_card_width(
    team_anchors,
    image_width
):
    """
    Estimate team-card width from spacing between detected headings.
    """

    x_differences = []

    visual_rows = group_anchors_into_visual_rows(
        team_anchors
    )

    for row in visual_rows:
        anchors = row["anchors"]

        for index in range(
            len(anchors) - 1
        ):
            difference = (
                anchors[index + 1]["x_centre"]
                - anchors[index]["x_centre"]
            )

            if difference > 100:
                x_differences.append(
                    difference
                )

    if x_differences:
        return int(
            median(
                x_differences
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
    values,
    tolerance
):
    """
    Convert nearby coordinate values into grid positions.
    """

    if not values:
        return []

    clusters = []

    for value in sorted(
        values
    ):
        matching_cluster = None

        for cluster in clusters:
            if (
                abs(
                    value
                    - cluster["average"]
                )
                <= tolerance
            ):
                matching_cluster = cluster
                break

        if matching_cluster is None:
            clusters.append({
                "average": value,
                "values": [value]
            })

        else:
            matching_cluster["values"].append(
                value
            )

            matching_cluster["average"] = int(
                sum(
                    matching_cluster["values"]
                )
                / len(
                    matching_cluster["values"]
                )
            )

    return sorted(
        cluster["average"]
        for cluster
        in clusters
    )


def find_anchor_near_position(
    team_anchors,
    expected_x,
    expected_y,
    x_tolerance,
    y_tolerance
):
    """
    Check whether a detected heading already occupies a grid position.
    """

    for anchor in team_anchors:
        if (
            abs(
                anchor["x_centre"]
                - expected_x
            )
            <= x_tolerance
            and abs(
                anchor["y_centre"]
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
    expected_x,
    expected_y,
    ocr_items,
    card_width
):
    """
    Find a placement number close to an expected card-heading position.
    """

    possible_ranks = []

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
            - item["x_centre"]
        )

        vertical_distance = abs(
            expected_y
            - item["y_centre"]
        )

        if (
            card_width * 0.30
            <= horizontal_distance
            <= card_width * 0.95
            and vertical_distance <= 60
        ):
            possible_ranks.append({
                "rank": number,
                "distance": (
                    horizontal_distance
                    + vertical_distance
                )
            })

    if not possible_ranks:
        return None

    best_match = min(
        possible_ranks,
        key=lambda match: match["distance"]
    )

    return best_match["rank"]


def find_nearby_rank(
    anchor,
    ocr_items,
    card_width
):
    """
    Find the placement number to the left of a detected heading.
    """

    return find_nearby_rank_for_position(
        expected_x=anchor["x_centre"],
        expected_y=anchor["y_centre"],
        ocr_items=ocr_items,
        card_width=card_width
    )


def infer_missing_ranks(
    ordered_anchors
):
    """
    Fill missing placement values where surrounding ranks make the
    missing value clear.
    """

    known_offsets = []

    for index, anchor in enumerate(
        ordered_anchors
    ):
        if anchor["placement"] is not None:
            known_offsets.append(
                anchor["placement"]
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
        if anchor["placement"] is None:
            inferred_rank = (
                index
                + likely_offset
            )

            if inferred_rank > 0:
                anchor["placement"] = inferred_rank
                anchor["placement_inferred"] = True

    return ordered_anchors


# ============================================================
# TARGETED TEAM-HEADING OCR
# ============================================================

def collect_team_heading_candidates(
    reader,
    source_image,
    expected_x,
    expected_y,
    card_width
):
    """
    Run targeted OCR around one expected heading position.
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

    possible_headings = []

    for pass_name, variant in create_retry_variants(
        heading_region
    ):
        retry_results = reader.readtext(
            variant,
            detail=1,
            paragraph=False,
            text_threshold=0.35,
            low_text=0.20,
            link_threshold=0.20,
            contrast_ths=0.05,
            adjust_contrast=0.80
        )

        for box, text, confidence in retry_results:
            corrected_team = normalise_team_heading(
                text
            )

            if corrected_team is None:
                continue

            coordinates = calculate_box_coordinates(
                box
            )

            possible_headings.append({
                "team": corrected_team,
                "original_team_ocr_text": text,
                "team_heading_confidence": round(
                    float(
                        confidence
                    ),
                    3
                ),
                "x_centre": (
                    coordinates["x_centre"]
                    + x_left
                ),
                "y_centre": (
                    coordinates["y_centre"]
                    + y_top
                ),
                "ocr_pass": pass_name
            })

    return possible_headings


def choose_best_team_heading_candidate(
    candidates,
    fallback=None
):
    """
    Choose the strongest team heading across multiple retry passes.

    Repeated readings receive additional weight.
    """

    candidates = list(
        candidates
    )

    if fallback is not None:
        candidates.append(
            fallback
        )

    if not candidates:
        return None

    grouped = {}

    for candidate in candidates:
        grouped.setdefault(
            candidate["team"],
            []
        ).append(
            candidate
        )

    scored_candidates = []

    for team_name, team_candidates in grouped.items():
        best_candidate = max(
            team_candidates,
            key=lambda item: item[
                "team_heading_confidence"
            ]
        )

        score = sum(
            item["team_heading_confidence"]
            for item
            in team_candidates
        )

        score += (
            0.15
            * len(
                team_candidates
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

        scored_candidates.append(
            (
                score,
                best_candidate
            )
        )

    return max(
        scored_candidates,
        key=lambda item: item[0]
    )[1]


def refine_detected_team_headers(
    team_anchors,
    reader,
    source_image,
    image_width
):
    """
    Recheck each detected heading.

    This helps where the full-image pass reads TEAM12 as TEAM2.
    """

    if not team_anchors:
        return []

    card_width = estimate_card_width(
        team_anchors,
        image_width
    )

    refined_anchors = []

    for anchor in team_anchors:
        candidates = collect_team_heading_candidates(
            reader=reader,
            source_image=source_image,
            expected_x=anchor["x_centre"],
            expected_y=anchor["y_centre"],
            card_width=card_width
        )

        best_candidate = choose_best_team_heading_candidate(
            candidates=candidates,
            fallback=anchor
        )

        best_candidate["header_refined"] = (
            best_candidate["team"]
            != anchor["team"]
        )

        refined_anchors.append(
            best_candidate
        )

    return refined_anchors


def recover_missing_team_headers(
    team_anchors,
    reader,
    source_image,
    ocr_items,
    image_width
):
    """
    Recover empty positions in the visible scoreboard grid.

    Example:
        Placement 8 contains TEAM24, but the initial OCR pass fails to
        read TEAM24. The expected middle position in the last grid row
        is identified and scanned again.

    Where the heading remains unreadable but the rank is visible, a
    placeholder team is returned instead of silently dropping the card.
    """

    if len(team_anchors) < 4:
        return team_anchors

    card_width = estimate_card_width(
        team_anchors,
        image_width
    )

    x_positions = cluster_coordinate_values(
        [
            anchor["x_centre"]
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
            anchor["y_centre"]
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
        len(x_positions) < 2
        or len(y_positions) < 2
    ):
        return team_anchors

    recovered_anchors = list(
        team_anchors
    )

    for expected_y in y_positions:
        for expected_x in x_positions:
            existing_anchor = find_anchor_near_position(
                team_anchors=recovered_anchors,
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

            if existing_anchor is not None:
                continue

            candidates = collect_team_heading_candidates(
                reader=reader,
                source_image=source_image,
                expected_x=expected_x,
                expected_y=expected_y,
                card_width=card_width
            )

            best_candidate = choose_best_team_heading_candidate(
                candidates=candidates
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

                print(
                    f"Recovered missing heading: "
                    f"{best_candidate['team']} "
                    f"from OCR text "
                    f"'{best_candidate['original_team_ocr_text']}'"
                )

                continue

            nearby_rank = find_nearby_rank_for_position(
                expected_x=expected_x,
                expected_y=expected_y,
                ocr_items=ocr_items,
                card_width=card_width
            )

            if nearby_rank is not None:
                placeholder_anchor = {
                    "team": (
                        f"UNKNOWN_TEAM_AT_PLACE_"
                        f"{nearby_rank}"
                    ),
                    "original_team_ocr_text": "",
                    "team_heading_confidence": 0.0,
                    "x_centre": expected_x,
                    "y_centre": expected_y,
                    "recovered_from_missing_grid_position": True,
                    "header_refined": False,
                    "team_name_needs_review": True
                }

                recovered_anchors.append(
                    placeholder_anchor
                )

                print(
                    f"Recovered missing card at placement "
                    f"{nearby_rank}, but its heading still "
                    f"needs review."
                )

    return recovered_anchors


def remove_duplicate_team_anchors(
    team_anchors
):
    """
    Remove multiple headings detected at approximately the same position.
    """

    unique_anchors = []

    for anchor in sorted(
        team_anchors,
        key=lambda item: (
            item["y_centre"],
            item["x_centre"]
        )
    ):
        duplicate_index = None

        for index, existing in enumerate(
            unique_anchors
        ):
            if (
                abs(
                    existing["x_centre"]
                    - anchor["x_centre"]
                )
                < 35
                and abs(
                    existing["y_centre"]
                    - anchor["y_centre"]
                )
                < 35
            ):
                duplicate_index = index
                break

        if duplicate_index is None:
            unique_anchors.append(
                anchor
            )

        else:
            existing = unique_anchors[
                duplicate_index
            ]

            if (
                anchor.get(
                    "team_heading_confidence",
                    0
                )
                > existing.get(
                    "team_heading_confidence",
                    0
                )
            ):
                unique_anchors[
                    duplicate_index
                ] = anchor

    return unique_anchors


# ============================================================
# DYNAMIC CARD REGIONS
# ============================================================

def build_team_regions(
    team_anchors,
    ocr_items,
    image_width,
    image_height
):
    """
    Create card areas dynamically from team-heading positions.
    """

    visual_rows = group_anchors_into_visual_rows(
        team_anchors
    )

    card_width = estimate_card_width(
        team_anchors,
        image_width
    )

    ordered_anchors = []

    for row_index, row in enumerate(
        visual_rows
    ):
        for column_index, anchor in enumerate(
            row["anchors"]
        ):
            anchor["visual_row"] = row_index
            anchor["visual_column"] = column_index

            ordered_anchors.append(
                anchor
            )

    for anchor in ordered_anchors:
        anchor["placement"] = find_nearby_rank(
            anchor=anchor,
            ocr_items=ocr_items,
            card_width=card_width
        )

        anchor["placement_inferred"] = False

    ordered_anchors = infer_missing_ranks(
        ordered_anchors
    )

    row_header_positions = [
        row["average_y"]
        for row
        in visual_rows
    ]

    for anchor in ordered_anchors:
        row_index = anchor["visual_row"]

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

        anchor["region"] = {
            "x_left": max(
                0,
                int(
                    anchor["x_centre"]
                    - card_width * 0.92
                )
            ),
            "x_right": min(
                image_width,
                int(
                    anchor["x_centre"]
                    + card_width * 0.12
                )
            ),
            "y_top": max(
                0,
                int(
                    anchor["y_centre"]
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
    items
):
    """
    Choose the best non-overlapping OCR blocks for a player name.

    This keeps split names such as:
        MR_ICE + MAN

    while avoiding duplicate retry readings such as:
        ChatGPT + ChatGPT
    """

    if not items:
        return []

    sorted_items = sorted(
        items,
        key=lambda item: (
            item["x_right"],
            item["x_left"]
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
                ]["x_right"]
                < item["x_left"] - 3
            ):
                compatible_index = earlier_index
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
            item["x_right"]
            - item["x_left"]
        )

        item_score = (
            width
            * max(
                0.15,
                item["confidence"]
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
            best_scores[index] = include_score

            selections[index] = (
                selections[
                    previous_index
                ]
                + [item]
            )

        else:
            best_scores[index] = exclude_score

            selections[index] = selections[
                index - 1
            ]

    return sorted(
        selections[-1],
        key=lambda item: item["x_left"]
    )


def extract_players_for_team(
    anchor,
    ocr_items,
    card_width
):
    """
    Extract player names and kills from one dynamically calculated card.
    """

    region = anchor["region"]

    region_items = []

    for item in ocr_items:
        if (
            region["x_left"]
            <= item["x_centre"]
            <= region["x_right"]
            and region["y_top"]
            <= item["y_centre"]
            <= region["y_bottom"]
        ):
            region_items.append(
                item
            )

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
        row_items = row["items"]

        numeric_candidates = []

        for item in row_items:
            number = parse_kill_number(
                item["text"]
            )

            if number is None:
                continue

            distance_from_kill_column = abs(
                item["x_centre"]
                - anchor["x_centre"]
            )

            if (
                distance_from_kill_column
                <= card_width * 0.22
            ):
                numeric_candidates.append({
                    "item": item,
                    "number": number,
                    "distance": (
                        distance_from_kill_column
                    )
                })

        if not numeric_candidates:
            continue

        kill_match = min(
            numeric_candidates,
            key=lambda candidate: (
                candidate["distance"],
                -candidate["item"][
                    "confidence"
                ]
            )
        )

        kill_item = kill_match["item"]
        kills = kill_match["number"]

        possible_name_items = [
            item
            for item
            in row_items
            if (
                item["x_right"]
                < kill_item["x_left"] - 4
                and parse_small_number(
                    item["text"]
                )
                is None
                and normalise_team_heading(
                    item["text"]
                )
                is None
            )
        ]

        name_items = weighted_non_overlapping_name_items(
            possible_name_items
        )

        if not name_items:
            continue

        raw_name = " ".join(
            item["text"].strip()
            for item
            in name_items
            if item["text"].strip()
        ).strip()

        if not raw_name:
            continue

        name_confidences = [
            item["confidence"]
            for item
            in name_items
        ]

        average_name_confidence = round(
            sum(
                name_confidences
            )
            / len(
                name_confidences
            ),
            3
        )

        players.append({
            "ign": raw_name,
            "kills": kills,
            "name_confidence": (
                average_name_confidence
            ),
            "kill_confidence": (
                kill_item["confidence"]
            ),
            "needs_review": (
                average_name_confidence
                < 0.70
            )
        })

    return players


# ============================================================
# DYNAMIC PLAYER-ROW RETRY
# ============================================================

def retry_team_region(
    anchor,
    reader,
    source_image,
    ocr_items
):
    """
    Retry OCR inside one dynamically calculated card region.
    """

    region = anchor["region"]

    x_left = region["x_left"]
    x_right = region["x_right"]
    y_top = region["y_top"]
    y_bottom = region["y_bottom"]

    image_region = source_image[
        y_top:y_bottom,
        x_left:x_right
    ]

    if image_region.size == 0:
        return

    for pass_name, variant in create_retry_variants(
        image_region
    ):
        retry_results = reader.readtext(
            variant,
            detail=1,
            paragraph=False,
            text_threshold=0.45,
            low_text=0.25,
            link_threshold=0.25,
            contrast_ths=0.05,
            adjust_contrast=0.70
        )

        append_raw_ocr_results(
            ocr_items=ocr_items,
            raw_results=retry_results,
            pass_name=pass_name,
            x_offset=x_left,
            y_offset=y_top
        )


# ============================================================
# LOAD IMAGE
# ============================================================

image_path = Path(
    IMAGE_PATH
)

if not image_path.exists():
    raise FileNotFoundError(
        f"Could not find '{IMAGE_PATH}'. "
        f"Place the screenshot in the same folder as test.py "
        f"or update IMAGE_PATH at the top of the script."
    )

image = cv2.imread(
    str(
        image_path
    )
)

if image is None:
    raise ValueError(
        f"OpenCV could not read '{IMAGE_PATH}'. "
        f"Confirm that it is a valid image file."
    )

image = cv2.resize(
    image,
    None,
    fx=UPSCALE_FACTOR,
    fy=UPSCALE_FACTOR,
    interpolation=cv2.INTER_CUBIC
)

image_height, image_width = (
    image.shape[:2]
)


# ============================================================
# INITIAL OCR PASS
# ============================================================

print(
    "Starting full-image OCR..."
)

reader = easyocr.Reader(
    ["en"],
    gpu=USE_GPU
)

raw_results = reader.readtext(
    image,
    detail=1,
    paragraph=False
)

ocr_items = []

append_raw_ocr_results(
    ocr_items=ocr_items,
    raw_results=raw_results,
    pass_name="full_image"
)


# ============================================================
# DETECT TEAM HEADINGS
# ============================================================

team_anchors = []

for item in ocr_items:
    corrected_team = normalise_team_heading(
        item["text"]
    )

    if corrected_team is None:
        continue

    team_anchors.append({
        "team": corrected_team,
        "original_team_ocr_text": item["text"],
        "team_heading_confidence": item[
            "confidence"
        ],
        "x_centre": item["x_centre"],
        "y_centre": item["y_centre"],
        "header_refined": False
    })

team_anchors = remove_duplicate_team_anchors(
    team_anchors
)

if not team_anchors:
    raise ValueError(
        "No team headings were detected. "
        "Try a clearer screenshot or increase UPSCALE_FACTOR."
    )

print(
    f"Detected {len(team_anchors)} team heading(s) "
    f"during the initial OCR pass."
)


# ============================================================
# REFINE TEAM HEADINGS
# ============================================================

team_anchors = refine_detected_team_headers(
    team_anchors=team_anchors,
    reader=reader,
    source_image=image,
    image_width=image_width
)

team_anchors = remove_duplicate_team_anchors(
    team_anchors
)


# ============================================================
# RECOVER MISSING TEAM CARDS
# ============================================================

team_anchors = recover_missing_team_headers(
    team_anchors=team_anchors,
    reader=reader,
    source_image=image,
    ocr_items=ocr_items,
    image_width=image_width
)

team_anchors = remove_duplicate_team_anchors(
    team_anchors
)

print(
    f"Using {len(team_anchors)} team heading(s) "
    f"after missing-card recovery."
)


# ============================================================
# BUILD DYNAMIC TEAM REGIONS
# ============================================================

team_regions, estimated_card_width = build_team_regions(
    team_anchors=team_anchors,
    ocr_items=ocr_items,
    image_width=image_width,
    image_height=image_height
)


# ============================================================
# EXTRACT PLAYER ROWS
# ============================================================

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

        print(
            f"Retrying OCR for {anchor['team']} "
            f"because only {initial_player_count} "
            f"player row(s) were detected..."
        )

        retry_team_region(
            anchor=anchor,
            reader=reader,
            source_image=image,
            ocr_items=ocr_items
        )

        players = extract_players_for_team(
            anchor=anchor,
            ocr_items=ocr_items,
            card_width=estimated_card_width
        )

    row_count_warning = (
        len(
            players
        )
        < EXPECTED_PLAYERS_PER_TEAM
    )

    structured_teams.append({
        "placement": anchor["placement"],
        "placement_inferred": anchor[
            "placement_inferred"
        ],
        "team": anchor["team"],
        "original_team_ocr_text": anchor.get(
            "original_team_ocr_text",
            ""
        ),
        "team_heading_confidence": anchor.get(
            "team_heading_confidence",
            0.0
        ),
        "team_name_needs_review": anchor.get(
            "team_name_needs_review",
            False
        ),
        "header_refined": anchor.get(
            "header_refined",
            False
        ),
        "recovered_from_missing_grid_position": anchor.get(
            "recovered_from_missing_grid_position",
            False
        ),
        "initial_player_count": initial_player_count,
        "detected_player_count": len(
            players
        ),
        "expected_player_count": (
            EXPECTED_PLAYERS_PER_TEAM
        ),
        "dynamic_retry_used": dynamic_retry_used,
        "row_count_warning": row_count_warning,
        "players": players
    })

structured_teams = sorted(
    structured_teams,
    key=lambda team: (
        team["placement"] is None,
        (
            team["placement"]
            if team["placement"] is not None
            else 999
        )
    )
)


# ============================================================
# SAVE JSON OUTPUT
# ============================================================

output = {
    "source_image": IMAGE_PATH,
    "image_width_after_upscaling": image_width,
    "image_height_after_upscaling": image_height,
    "estimated_card_width": estimated_card_width,
    "detected_team_count": len(
        structured_teams
    ),
    "teams": structured_teams
}

with open(
    OUTPUT_JSON_PATH,
    "w",
    encoding="utf-8"
) as json_file:
    json.dump(
        output,
        json_file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# SAVE CSV OUTPUT
# ============================================================

with open(
    OUTPUT_CSV_PATH,
    "w",
    newline="",
    encoding="utf-8-sig"
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "placement",
            "placement_inferred",
            "team",
            "team_name_needs_review",
            "ign",
            "kills",
            "name_confidence",
            "kill_confidence",
            "needs_review",
            "dynamic_retry_used",
            "row_count_warning"
        ]
    )

    writer.writeheader()

    for team in structured_teams:
        for player in team["players"]:
            writer.writerow({
                "placement": team[
                    "placement"
                ],
                "placement_inferred": team[
                    "placement_inferred"
                ],
                "team": team[
                    "team"
                ],
                "team_name_needs_review": team[
                    "team_name_needs_review"
                ],
                "ign": player[
                    "ign"
                ],
                "kills": player[
                    "kills"
                ],
                "name_confidence": player[
                    "name_confidence"
                ],
                "kill_confidence": player[
                    "kill_confidence"
                ],
                "needs_review": player[
                    "needs_review"
                ],
                "dynamic_retry_used": team[
                    "dynamic_retry_used"
                ],
                "row_count_warning": team[
                    "row_count_warning"
                ]
            })


# ============================================================
# SAVE DEBUG OUTPUT
# ============================================================

debug_output = {
    "detected_team_regions": team_regions,
    "all_ocr_items": ocr_items
}

with open(
    DEBUG_JSON_PATH,
    "w",
    encoding="utf-8"
) as debug_file:
    json.dump(
        debug_output,
        debug_file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# PRINT RESULT
# ============================================================

print(
    "\nOCR completed successfully.\n"
)

print(
    json.dumps(
        output,
        indent=4,
        ensure_ascii=False
    )
)

print(
    "\nFiles created:"
)

print(
    f"  - {OUTPUT_JSON_PATH}"
)

print(
    f"  - {OUTPUT_CSV_PATH}"
)

print(
    f"  - {DEBUG_JSON_PATH}"
)