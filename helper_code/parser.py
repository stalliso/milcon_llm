# Code with all functions necessary to parse our Project Data Sheets

import fitz  # pymupdf
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import os
from pprint import pprint
import re

# Create custom data class that will represent pdf text boxes and custom parser object to extract those text boxes:
#________________________________________________________________________________________________________________________
@dataclass
class TextBox:
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    


class FormTextExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)

    def extract_boxes(self) -> List[TextBox]:
        boxes: List[TextBox] = []
        for page_index in range(len(self.doc)):
            page = self.doc[page_index]
            for b in page.get_text("blocks"):
                x0, y0, x1, y1, text, *_ = b
                txt = (text or "").strip()
                if not txt:
                    continue
                boxes.append(
                    TextBox(
                        page=page_index,
                        x0=float(x0),
                        y0=float(y0),
                        x1=float(x1),
                        y1=float(y1),
                        text=txt,
                    )
                )

        # Visual flow: page, then y (top→bottom), then x (left→right)
        boxes.sort(key=lambda b: (b.page, b.y0, b.x0))
        return boxes

    def extract_as_dicts(self) -> List[Dict[str, Any]]:
        return [asdict(b) for b in self.extract_boxes()]
#________________________________________________________________________________________________________________________




# Functions to obtain Metadata/General Project Info from a pdf that has been extracted into boxes with the parser:
#________________________________________________________________________________________________________________________
def get_proj(boxes:list)->str: # Project i.e. "P930"
    text = boxes[1].text 
    return text.splitlines()[2].lstrip()

def get_title(boxes:list)->str: # Title i.e. "Galley II"
    text = boxes[1].text
    return text.splitlines()[1].lstrip()

def get_inst(boxes:list)->str: # Installation
    text = boxes[6].text
    return text.splitlines()[0].lstrip()

def get_CWE(boxes:list)->int: # Current Estimate
    idx = 0
    for i in range(10):
        if "CWE($K)".casefold() in boxes[i].text.casefold(): # Find line based on field name
            idx = i
    text = boxes[idx].text
    return int(text.replace("CWE($K):", "").replace(",", "").strip())

def get_CCN(boxes:list) ->int: # Control Number
    idx = 0
    for i in range(10):
        if "CCN".casefold() in boxes[i].text.casefold(): # Find line based on field name
            idx = i
    text = boxes[idx].text
    return int(text.replace("Project CCN:", "").strip())

def get_regn(boxes:list) ->str: # Region
    text = boxes[9].text
    return text.splitlines()[0]

def get_lead_propt(boxes: list) -> str: # Lead proponent
    text = boxes[9].text
    lines = text.splitlines()
    
    # Safely grab the 3rd line if it exists
    if len(lines) > 2:
        return lines[2].strip()
    # Fallback: if there are only 2 lines, the data might be on the second line
    elif len(lines) == 2:
        return lines[1].strip()
    # If the box is basically empty or just contains the header, return nothing
    else:
        return None

def get_cocom(boxes:list)->str: # COCOM
    text = boxes[8].text
    return text.replace("COCOM:","").strip()

def get_fac_msn(boxes:list)->str: # Facility mission
    text = boxes[10].text
    return "".join(text.splitlines()[1:]).lstrip()

def get_scope(boxes:list)->str: # Project Scope
    text = boxes[12].text
    return "".join(text.splitlines())

def get_imp(boxes:list)->str: # Impact if not provided
    text = boxes[14].text
    return "".join(text.splitlines())
#________________________________________________________________________________________________________________________




# Functions to extract scores and descriptions from the descriptive fields:
#________________________________________________________________________________________________________________________
# For this portion of each data sheet, some of the data sheets have the entire right side (the lead proponent scores and descriptions) blank.
# The parser puts the regional and lead proponent scores on the same line which is easy to handle. However, it places the regional and 
# lead proponent scores on separate lines. This means we cannot write functions to pull from a particular line because a fully-filled form
# will have more lines than the ones that leave lead proponent sections blank. Functions therefore need to find the line number that contains
# a field name such as "lead proponent mission alignment" and use indices in relation to that line number since we cannot use raw line numbers.

# =============================================================================
#  SAFE EXTRACTION HELPER
# =============================================================================

def _safe_extract_field(score_text: str, desc_text: str, is_lead: bool = False, split_desc: bool = False) -> dict:
    """
    Universally and safely extracts the score and description, catching missing lines
    and format errors without crashing the script.
    """
    score_lines = score_text.splitlines()
    desc_lines = desc_text.splitlines()
    
    # If this is a Lead Proponent check, and there is no second line in the score box,
    # it means the Lead Proponent left their section entirely blank.
    if is_lead and len(score_lines) <= 1:
        return {"description": None, "score": None}
        
    score = None
    # Extract Score
    if score_lines:
        # Region gets line 0. Lead Proponent gets line 1.
        s_idx = 1 if is_lead else 0
        try:
            score = int(score_lines[s_idx].strip())
        except ValueError:
            score = None  # Prevents crash if the box grabbed text instead of a number
            
    description = None
    # Extract Description
    if desc_lines:
        # Special case for Lead Proponent Operational Cost (where desc is on the 2nd line of the same box)
        if split_desc and is_lead and len(desc_lines) > 1:
            description = " ".join(desc_lines[1:]).strip()
        else:
            description = " ".join(desc_lines).strip()
            
    return {"description": description, "score": score}


# =============================================================================
#  FIELD EXTRACTION FUNCTIONS
# =============================================================================

def regn_mis_algn(boxes: list) -> dict: # Regional Mission Alignment
    idx = 0
    for i in range(15, 21):
        if boxes[i].text.casefold() == 'Lead Proponent Mission Alignment:\nRegion Mission Alignment:'.casefold():
            idx = i
            break
    return _safe_extract_field(score_text=boxes[idx + 1].text, desc_text=boxes[idx + 2].text, is_lead=False)


def ld_propt_mis_algn(boxes: list) -> dict: # Lead Proponent Mission Alignment
    idx = 0
    for i in range(15, 21):
        if boxes[i].text.casefold() == 'Lead Proponent Mission Alignment:\nRegion Mission Alignment:'.casefold():
            idx = i
            break
    return _safe_extract_field(score_text=boxes[idx + 1].text, desc_text=boxes[idx + 3].text, is_lead=True)


def regn_rd_spt(boxes: list) -> dict: # Region Readiness Support
    idx = 0
    for i in range(16, 22):
        if boxes[i].text.casefold() == 'Lead Proponent Readiness Support:\nRegion Readiness Support:'.casefold():
            idx = i
            break
    return _safe_extract_field(score_text=boxes[idx + 1].text, desc_text=boxes[idx + 2].text, is_lead=False)


def ld_propt_rd_spt(boxes: list) -> dict: # Lead proponent readiness support
    idx = 0
    for i in range(16, 22):
        if boxes[i].text.casefold() == 'Lead Proponent Readiness Support:\nRegion Readiness Support:'.casefold():
            idx = i
            break
    return _safe_extract_field(score_text=boxes[idx + 1].text, desc_text=boxes[idx + 3].text, is_lead=True)


def regn_op_cost(boxes: list) -> dict: # Region Operational Cost
    idx = 0
    for i in range(20, 27):
        if boxes[i].text.casefold() == 'Lead Proponent Operational Cost:\nRegion Operational Cost:'.casefold():
            idx = i
            break
    return _safe_extract_field(score_text=boxes[idx + 1].text, desc_text=boxes[idx + 2].text, is_lead=False)


def ld_propt_op_cost(boxes: list) -> dict: # Lead proponent operational cost
    idx = 0
    for i in range(20, 27):
        if boxes[i].text.casefold() == 'Lead Proponent Operational Cost:\nRegion Operational Cost:'.casefold():
            idx = i
            break
    # Note: split_desc=True because Lead Prop Op Cost shares the desc_text box
    return _safe_extract_field(score_text=boxes[idx + 1].text, desc_text=boxes[idx + 2].text, is_lead=True, split_desc=True)


def regn_sev_st(boxes: list) -> dict: # Region severity statement
    idx = 0
    for i in range(23, 30):
        if boxes[i].text.casefold() == 'Lead Proponent Severity Statement:\nRegion Severity Statement:'.casefold():
            idx = i
            break
    return _safe_extract_field(score_text=boxes[idx + 1].text, desc_text=boxes[idx + 2].text, is_lead=False)


def ld_propt_sev_st(boxes: list) -> dict: # Lead proponent severity statement
    idx = 0
    for i in range(23, 30):
        if boxes[i].text.casefold() == 'Lead Proponent Severity Statement:\nRegion Severity Statement:'.casefold():
            idx = i
            break
    return _safe_extract_field(score_text=boxes[idx + 1].text, desc_text=boxes[idx + 3].text, is_lead=True)


def regn_urg_st(boxes: list) -> dict: # Region urgency statement
    idx = 0
    for i in range(30, 40):
        if boxes[i].text.casefold() == 'Lead Proponent Urgency Statement:\nRegion Urgency Statement:'.casefold():
            idx = i
            break
    return _safe_extract_field(score_text=boxes[idx + 1].text, desc_text=boxes[idx + 2].text, is_lead=False)


def ld_propt_urg_st(boxes: list) -> dict: # Lead proponent urgency statement
    idx = 0
    for i in range(30, 40):
        if boxes[i].text.casefold() == 'Lead Proponent Urgency Statement:\nRegion Urgency Statement:'.casefold():
            idx = i
            break
    return _safe_extract_field(score_text=boxes[idx + 1].text, desc_text=boxes[idx + 3].text, is_lead=True)
#________________________________________________________________________________________________________________________



# Get the region/lead proponent metrics (RAC, ROI, PCI)
#________________________________________________________________________________________________________________________
# These are tricky and require some complex handling because all three appear on a single line of text or in a single text box. 
# Sometimes, within the textbox, the value precedes the field name it corresponds to, sometimes it succeeds it, sometimes there is 
# extra whitespace between the field name and value and sometimes they are left entirely blank.

def rac_roi_pci(boxes: list)->dict:

    fields = ["RAC:", "ROI:", "PCI:"]
    results = {"RAC":None, "ROI":None, "PCI":None}

    idx = 0
    for i in range(35,46):
        if "RAC:" in boxes[i].text:
            tokens = [t.strip() for t in boxes[i].text.replace("\n", " ").split() if t.strip()]

            current_field = None
            pending_value = None
            for tok in tokens:
                if tok in fields:

                    if pending_value is not None:
                        results[tok[:-1]] = pending_value
                        pending_value = None
                    current_field = tok[:-1] # Get rid of colon

                else:
                    if current_field is None:
                        pending_value = tok
                    else:
                        results[current_field] = tok
                        current_field = None
    return results
#________________________________________________________________________________________________________________________


# Extract the facility information tables, these are extremely difficult because they have differing numbers of rows and 
# their layouts differ.
#________________________________________________________________________________________________________________________

# Known column header tokens that appear in the table (for single-block detection)
facility_table_headers = frozenset({
    "facility id", "predom", "ccn", "facility no.", "mdi", "prv ($k)", "um",
    "work", "type", "quantity", "cond", "rtg", "conf", "yr blt",
    "rpa name", "mtd"
})

# Exact output shape (from project930/942 expected output)
facility_row_keys = [
    "Facility ID", "Facility No.", "RPA Name", "Predom CCN", "PRV ($K)", "Work Type",
    "Quantity", "UM", "MDI", "Cond Rtg", "Conf Rtg", "MTD", "Yr Blt",
]
# Single-row: main block has 9 data values in this order (indices 0..8)
single_row_data_order = [
    "Facility ID", "Predom CCN", "MDI", "PRV ($K)", "UM", "Work Type", "Quantity", "Cond Rtg", "Conf Rtg",
]
# Extended for forms that put data first then headers, with Facility No. and Yr Blt (e.g. RM23-0513)
single_row_data_order_extended = [
    "Facility ID", "Predom CCN", "MDI", "PRV ($K)", "UM", "Work Type", "Quantity",
    "Facility No.", "Cond Rtg", "Conf Rtg", "Yr Blt",
]
# Multi-row: 10 cells per row -> column index to key
multi_row_cell_order = [
    "Facility ID", "RPA Name", "Predom CCN", "PRV ($K)", "Work Type", "Quantity", "UM", "MDI", "Cond Rtg", "Conf Rtg",
]

def y_cluster_tolerance() -> float:
    return 3.0  # points; boxes on same row have y0 within this

def find_facility_info_region(boxes: List[TextBox]) -> Optional[tuple]:
    """Find (page, y_start, y_end) for the Facility Information section, or None."""
    title_y0 = None
    page = None
    for b in boxes:
        if "Facility Information" in b.text and b.text.strip() == "Facility Information":
            page = b.page
            title_y0 = b.y0
            break
    if title_y0 is None or page is None:
        return None
    y_end = None
    for b in boxes:
        if b.page != page or b.y0 <= title_y0:
            continue
        if "PRV-Weighted" in b.text:
            y_end = b.y0
            break
    if y_end is None:
        y_end = 1e6
    return (page, title_y0, y_end)

def is_likely_header_cell(text: str) -> bool:
    t = text.strip().lower().replace("\n", " ")
    return "facility id" in t or "facility no." in t or "rpa name" in t

def get_region_boxes(boxes: List[TextBox], page: int, y_start: float, y_end: float) -> List[TextBox]:
    return [
        b for b in boxes
        if b.page == page and y_start < b.y0 < y_end
        and "Facility Information" not in b.text
        and "PRV-Weighted" not in b.text
    ]


def _line_looks_like_facility_data(ln: str) -> bool:
    """True if line looks like a data value (facility ID or number), not a header."""
    t = ln.strip()
    if not t or len(t) < 2:
        return False
    lower = t.lower()
    if lower in ("facility id", "facility no.", "rpa name", "predom", "ccn", "mdi", "prv ($k)", "um", "work", "type", "quantity", "cond", "rtg", "conf", "yr blt", "mtd"):
        return False
    if "facility" in lower or "predom" in lower or "rpa" in lower or "quantity" in lower:
        return False
    digits = "".join(c for c in t if c.isdigit())
    if len(digits) >= 4 and "," not in t:
        return True  # e.g. 121851, 9129.0
    if len(t) >= 6 and any(c.isalpha() for c in t) and any(c.isdigit() for c in t):
        return True  # e.g. NFA200000403234
    if len(t) >= 2 and t.replace(".", "").replace(",", "").isdigit():
        return True  # numeric with comma/period
    return False


def _find_data_first_facility_block(region_boxes: List[TextBox]) -> Optional[TextBox]:
    """Find a box that contains 'Facility ID' and has data lines before the header labels (data-first layout)."""
    for b in region_boxes:
        if "Facility ID" not in b.text and "facility id" not in b.text.lower():
            continue
        lines = [ln.strip() for ln in b.text.splitlines() if ln.strip()]
        seen_data = False
        for ln in lines:
            if _line_looks_like_facility_data(ln):
                seen_data = True
            if seen_data and (ln.lower() == "facility id" or "facility id" in ln.lower()):
                return b  # we saw data then hit header
    return None


def _collect_rpa_name_from_region_boxes(region_boxes: List[TextBox], main_block: TextBox) -> str:
    """Collect RPA Name from boxes below the main block (e.g. separate RPA Name / value cells)."""
    tol = y_cluster_tolerance()
    # Boxes strictly below the main block (or same y but we want to skip main)
    parts = []
    for b in region_boxes:
        if b is main_block or b.y0 <= main_block.y0 + tol:
            continue
        for ln in b.text.splitlines():
            t = ln.strip()
            if not t or t.lower() in ("rpa name", "mtd", "facility id"):
                continue
            if _is_header_like_cell(t):
                continue
            if len(t) <= 2 and t.isdigit():
                continue  # skip short numbers like 55 (Facility No.)
            if len(t) == 1:
                continue  # skip single chars (e.g. "R" from "R\n15KVOH\nMTD")
            parts.append(t)
    return " ".join(parts) if parts else ""


def parse_data_first_single_facility(region_boxes: List[TextBox]) -> List[Dict[str, str]]:
    """
    Parse when one block has DATA first then HEADERS (e.g. RM23-0513).
    Uses extended field order and collects RPA Name from other boxes.
    """
    main_block = _find_data_first_facility_block(region_boxes)
    if main_block is None:
        return []
    lines = [ln.strip() for ln in main_block.text.splitlines() if ln.strip()]
    header_set = facility_table_headers
    data_lines = []
    for ln in lines:
        token = ln.lower()
        if token in header_set or "facility id" in token or "facility no." in token or "predom" in token or "cond" in token or "conf" in token or token == "ccn" or token == "mdi" or "prv" in token or token == "um" or token == "work" or token == "type" or token == "quantity" or "rtg" in token or "yr blt" in token:
            break
        data_lines.append(ln)
    if not data_lines:
        return []
    result_row = {k: "" for k in facility_row_keys}
    order = single_row_data_order_extended
    for i, key in enumerate(order):
        if i < len(data_lines):
            result_row[key] = data_lines[i].strip()
    if len(data_lines) >= 9 and not result_row.get("Conf Rtg") and result_row.get("Cond Rtg"):
        result_row["Conf Rtg"] = result_row["Cond Rtg"]
    rpa = _collect_rpa_name_from_region_boxes(region_boxes, main_block)
    if rpa:
        result_row["RPA Name"] = rpa
    return [result_row]


def multi_row_parse(region_boxes: List[TextBox]) -> List[Dict[str, str]]:
    """Parse when table is laid out as separate boxes per cell (multiple rows)."""
    if not region_boxes:
        return []
    tol = y_cluster_tolerance()
    # Group by y0 (row)
    rows_y: List[float] = []
    for b in region_boxes:
        y = b.y0
        if not rows_y or abs(y - rows_y[-1]) > tol:
            rows_y.append(y)
    # Assign each box to row index by nearest row y
    def row_idx(b: TextBox) -> int:
        i = 0
        for i, ry in enumerate(rows_y):
            if abs(b.y0 - ry) <= tol:
                return i
            if b.y0 < ry:
                return i
        return len(rows_y) - 1
    by_row: Dict[int, List[TextBox]] = {}
    for b in region_boxes:
        r = row_idx(b)
        by_row.setdefault(r, []).append(b)
    # Sort rows by y; within each row sort by x0
    row_indices = sorted(by_row.keys(), key=lambda r: rows_y[r] if r < len(rows_y) else 0)
    table_rows: List[List[str]] = []
    for r in row_indices:
        cells = sorted(by_row[r], key=lambda b: b.x0)
        # Keep raw text (with newlines) so header row can be split into column names
        row_texts = [c.text.strip() for c in cells]
        table_rows.append(row_texts)
    if not table_rows:
        return []
    # First row that looks like headers
    header_row_idx = 0
    for i, row in enumerate(table_rows):
        if any(is_likely_header_cell(c) for c in row):
            header_row_idx = i
            break
    header_cells = table_rows[header_row_idx]
    if len(header_cells) == 1 and "\n" in header_cells[0]:
        tokens = [s.strip() for s in header_cells[0].splitlines() if s.strip()]
        headers = merge_header_tokens(tokens)
    else:
        headers = [c.strip() or f"Col{j}" for j, c in enumerate(header_cells)]
    data_rows = table_rows[header_row_idx + 1:]
    # Skip secondary header row that is only "RPA Name" and "MTD"
    if data_rows and len(data_rows[0]) <= 2:
        row0 = [cell.replace("\n", " ").strip().lower() for cell in data_rows[0]]
        if row0 == ["rpa name", "mtd"] or (len(row0) == 2 and "rpa name" in row0[0] and "mtd" in row0[1]):
            data_rows = data_rows[1:]
    if not data_rows:
        return []
    # Drop rows that are entirely header-like (repeated header or subheader)
    def row_is_header_like(r):
        return all(_is_header_like_cell(cell.replace("\n", " ").strip()) for cell in r)
    data_rows = [r for r in data_rows if not row_is_header_like(r)]
    if not data_rows:
        return []
    # Expand rows: if any cell has newlines, split into one row per line
    expanded_rows = expand_multiline_cells(data_rows)
    # Build row dicts using schema so Facility No. is blank and Predom CCN is one value
    result = []
    for row in expanded_rows:
        d = cells_to_facility_row(row, headers)
        result.append(d)
    return result

def merge_header_tokens(tokens: List[str]) -> List[str]:
    """Merge split header tokens: Predom+CCN, Work+Type, Cond+Rtg, Conf+Rtg."""
    merged = []
    i = 0
    while i < len(tokens):
        t = tokens[i].lower()
        if i + 1 < len(tokens) and t == "predom" and tokens[i + 1].lower() == "ccn":
            merged.append("Predom CCN"); i += 2
        elif i + 1 < len(tokens) and t == "work" and tokens[i + 1].lower() == "type":
            merged.append("Work Type"); i += 2
        elif i + 1 < len(tokens) and t == "cond" and tokens[i + 1].lower() == "rtg":
            merged.append("Cond Rtg"); i += 2
        elif i + 1 < len(tokens) and t == "conf" and tokens[i + 1].lower() == "rtg":
            merged.append("Conf Rtg"); i += 2
        else:
            merged.append(tokens[i]); i += 1
    return merged

def expand_multiline_cells(data_rows: List[List[str]]) -> List[List[str]]:
    """Expand rows where cells contain newlines into one row per line (one facility per row)."""
    out = []
    for row in data_rows:
        cell_lines = [c.splitlines() for c in row]
        n_lines = max(len(cl) for cl in cell_lines) if cell_lines else 0
        for k in range(n_lines):
            out.append([
                (cell_lines[j][k] if k < len(cell_lines[j]) else "").strip()
                for j in range(len(row))
            ])
    return out

def cells_to_facility_row(cells: List[str], headers: List[str]) -> Dict[str, str]:
    """Map 10 cells to row dict: 0=Facility ID, 1=RPA Name, 2=Predom CCN, 3=PRV ($K), 4=Work Type, 5=Quantity, 6=UM, 7=MDI, 8=Cond Rtg, 9=Conf Rtg; Facility No., MTD, Yr Blt blank."""
    d = {k: "" for k in facility_row_keys}
    for i, key in enumerate(multi_row_cell_order):
        if i < len(cells):
            d[key] = cells[i].replace("\n", " ").strip()
    return d


# -----------------------------------------------------------------------------
# Facility row validation (filter junk from inconsistent table layouts)
# -----------------------------------------------------------------------------

# Facility IDs are typically 5–6 digit numbers; row indices or page numbers are 1–2 digits
_FACILITY_ID_MIN_DIGITS = 4
# RPA Name should be descriptive text; numeric-only values are usually misplaced Quantity
_NUMERIC_ONLY_RE = re.compile(r"^[\d,\s.]+$")


def _looks_like_facility_id(value: str) -> bool:
    """True if value looks like a real facility ID (e.g. 121379), not a row number or quantity."""
    if not value or not value.strip():
        return False
    # Real IDs are plain digits (no comma); quantities are often "4,855" or "10,473"
    if "," in value:
        return False
    digits = "".join(c for c in value if c.isdigit())
    return len(digits) >= _FACILITY_ID_MIN_DIGITS


def _rpa_name_looks_numeric(value: str) -> bool:
    """True if RPA Name looks like a misplaced number (e.g. '3,880')."""
    if not value or not value.strip():
        return False
    return bool(_NUMERIC_ONLY_RE.match(value.strip()))


def _is_header_like_cell(text: str) -> bool:
    """True if cell text looks like a column header, not data."""
    t = text.strip().lower()
    if not t:
        return False
    header_like = (
        "facility id", "facility no.", "rpa name", "predom", "ccn", "prv ($k)",
        "work type", "quantity", "um", "mdi", "cond rtg", "conf rtg", "mtd", "yr blt",
    )
    return t in header_like or (len(t) < 20 and any(h in t for h in header_like))


def is_valid_facility_row(row: Dict[str, str]) -> bool:
    """
    Return False for rows that are clearly junk: misaligned data, row indices,
    mostly empty rows, or header-like content in data fields.
    """
    fid = (row.get("Facility ID") or "").strip()
    rpa = (row.get("RPA Name") or "").strip()
    non_empty_count = sum(1 for v in row.values() if (v or "").strip())

    # Reject almost entirely empty rows (e.g. only one small number)
    if non_empty_count <= 1 and (not fid or len(fid) <= 2):
        return False

    # Reject rows where Facility ID looks like a row index (1–3 digits)
    if fid and not _looks_like_facility_id(fid):
        # Allow row only if it has substantial other data and could be continuation (rare)
        if non_empty_count <= 2 and _rpa_name_looks_numeric(rpa):
            return False
        if non_empty_count <= 1:
            return False

    # RPA Name should not be purely numeric (misplaced Quantity)
    if _rpa_name_looks_numeric(rpa):
        return False

    # Reject rows that are entirely header-like tokens
    if rpa and _is_header_like_cell(rpa):
        return False
    if fid and _is_header_like_cell(fid):
        return False

    # Need at least one real identifier: valid Facility ID or non-empty text RPA Name
    has_valid_id = _looks_like_facility_id(fid)
    # RPA Name must be substantive (e.g. "HOUSING WAREHOUSE"); reject 1–2 char fragments like "ME"
    _RPA_NAME_MIN_LEN = 3
    has_real_rpa = (
        rpa
        and len(rpa) >= _RPA_NAME_MIN_LEN
        and not _rpa_name_looks_numeric(rpa)
        and not _is_header_like_cell(rpa)
    )
    if not has_valid_id and not has_real_rpa:
        return False

    return True


def filter_valid_facility_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Return only rows that pass is_valid_facility_row."""
    return [r for r in rows if is_valid_facility_row(r)]

def single_row_parse(region_boxes: List[TextBox]) -> List[Dict[str, str]]:
    """Parse when one block contains both header labels and one row of data (single facility)."""
    # Find the block that contains "Facility ID" (and similar)
    main_block = None
    rpa_name_val = None
    mtd_val = None
    for b in region_boxes:
        if "Facility ID" in b.text or "facility id" in b.text.lower():
            main_block = b
        elif "RPA Name" in b.text:
            lines = b.text.splitlines()
            if len(lines) >= 1 and lines[0].strip() and not lines[0].strip().lower().startswith("rpa"):
                rpa_name_val = lines[0].strip()
        elif b.text.strip() == "MTD" or (len(b.text.splitlines()) == 1 and b.text.strip() and b.text.strip() != "RPA Name"):
            t = b.text.strip()
            if t != "RPA Name" and t != "MTD":
                mtd_val = t
            elif t == "MTD":
                pass  # header only
    if main_block is None:
        return []
    lines = [ln.strip() for ln in main_block.text.splitlines() if ln.strip()]
    # Split into data lines (before headers) and header labels
    header_set = facility_table_headers
    data_lines = []
    header_lines = []
    seen_headers = False
    for ln in lines:
        token = ln.lower()
        if not seen_headers and (token in header_set or "facility id" in token or "facility no." in token or "predom" in token or "cond" in token or "conf" in token):
            seen_headers = True
        if seen_headers:
            header_lines.append(ln)
        else:
            data_lines.append(ln)

    # If no clear split, treat last N lines as headers using known list
    if not header_lines and data_lines:
        header_lines = data_lines[-14:]
        data_lines = data_lines[:-14]
    # Exact mapping: 9 data values in order Facility ID, Predom CCN, MDI, PRV ($K), UM, Work Type, Quantity, Cond Rtg, Conf Rtg
    result_row = {k: "" for k in facility_row_keys}
    for i, key in enumerate(single_row_data_order):
        if i < len(data_lines):
            result_row[key] = data_lines[i].strip()
    if rpa_name_val is not None:
        result_row["RPA Name"] = rpa_name_val
    if mtd_val is not None:
        result_row["MTD"] = mtd_val
    return [result_row]

def extract_facility_information_table(boxes: List[TextBox]) -> List[Dict[str, str]]:
    """
    Extract the 'Facility Information' table from project data sheet boxes.
    Works for both single-row (one facility) and multi-row (multiple facilities).
    Returns a list of dicts, one per facility row; keys are column headers.
    """
    region = find_facility_info_region(boxes)
    if not region:
        return []
    page, y_start, y_end = region
    region_boxes = get_region_boxes(boxes, page, y_start, y_end)
    if not region_boxes:
        return []
    # Prefer data-first single block when present (e.g. RM23-0513: one block with data then headers)
    if _find_data_first_facility_block(region_boxes) is not None:
        rows = parse_data_first_single_facility(region_boxes)
        return filter_valid_facility_rows(rows)
    # Detect layout: multi-row if we have several boxes at the same y0 (different x0)
    tol = y_cluster_tolerance()
    y_to_boxes: Dict[float, List[TextBox]] = {}
    for b in region_boxes:
        y = round(b.y0 / tol) * tol
        y_to_boxes.setdefault(y, []).append(b)
    same_y_multi = sum(1 for lst in y_to_boxes.values() if len(lst) > 1)
    multi_row_layout = same_y_multi >= 2  # at least 2 rows with multiple cells
    if multi_row_layout:
        rows = multi_row_parse(region_boxes)
    else:
        rows = single_row_parse(region_boxes)
    return filter_valid_facility_rows(rows)
#________________________________________________________________________________________________________________________





# Apply functions in concert to get a consolidated Prject Data Sheet
#________________________________________________________________________________________________________________________
def get_pds(boxes:list)->dict:
    pds ={
        get_proj(boxes):{

            # Top section of the form (metadata): 
            "title":get_title(boxes),
            "installation":get_inst(boxes),
            "CWE":get_CWE(boxes),
            "CCN":get_CCN(boxes),
            "region":get_regn(boxes),
            "lead_proponent":get_lead_propt(boxes),
            "COCOM":get_cocom(boxes),
            "scope":get_scope(boxes),
            "impact_if_not_provided":get_imp(boxes),

            # Scored fields:
            "region_mission_alignment":regn_mis_algn(boxes),
            "lead_proponent_mission_alignment":ld_propt_mis_algn(boxes),
            "region_readiness_support":regn_rd_spt(boxes),
            "lead_proponent_readiness_support":ld_propt_rd_spt(boxes),
            "region_operational_cost":regn_op_cost(boxes),
            "lead_proponent_operational_cost":ld_propt_op_cost(boxes),
            "region_severity_statement":regn_sev_st(boxes),
            "lead_proponent_severity_statement": ld_propt_sev_st(boxes),
            "region_urgency_statement":regn_urg_st(boxes),
            "lead_proponent_urgency_statement":ld_propt_urg_st(boxes),

            # Region/lead proponent metrics:
            "metrics":rac_roi_pci(boxes),

            # Facility Information table:
            "facility_information":extract_facility_information_table(boxes)
        }
    }
    return pds
#________________________________________________________________________________________________________________________


# Example usage:
'''
#________________________________________________________________________________________________________________________
files = ["docs/projects/P1616.pdf", "docs/projects/RM23-0513.pdf"]
cwd = os.getcwd()
parent = os.path.dirname(cwd)
file_paths = [os.path.join(parent, file) for file in files]

proj1616 = FormTextExtractor(file_paths[0]).extract_boxes()
projrm23_0513 = FormTextExtractor(file_paths[1]).extract_boxes()

proj1616_txt = get_pds(proj1616)
projrm23_0513_txt = get_pds(projrm23_0513)

pprint(proj1616_txt, width = 100)
pprint(projrm23_0513_txt, width = 100)

#________________________________________________________________________________________________________________________
'''







