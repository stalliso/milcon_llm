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

def get_lead_propt(boxes:list)->str: # Lead proponent
    text = boxes[9].text
    return text.splitlines()[2]

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

def regn_mis_algn(boxes:list)->dict: # Regional Mission Alignment Score and Description
    idx = 0
    for i in range(15,21):
        if boxes[i].text.casefold() == 'Lead Proponent Mission Alignment:\nRegion Mission Alignment:'.casefold(): # Find line based on field name
            idx = i

    text = boxes[idx + 2].text
    score = int(boxes[idx + 1].text.splitlines()[0].strip())
    description = "".join(text.splitlines()).lstrip()
    return {"description":description, "score":score}

def ld_propt_mis_algn(boxes:list)->dict: # Lead Proponent Mission Alignment Score and Description (Some have the lead propt side of the page blank)
    idx = 0
    for i in range(15,21):
        if boxes[i].text.casefold() == 'Lead Proponent Mission Alignment:\nRegion Mission Alignment:'.casefold(): # Find line based on field name
            idx = i
    text = boxes[idx + 3].text
    if len(boxes[idx + 1].text.splitlines())>1:
        score = int(boxes[16].text.splitlines()[1].strip())
        description = "".join(text.splitlines()).lstrip()
        return {"description":description, "score":score}
    else:
        return {"description": None, "score":None}

def regn_rd_spt(boxes:list)->dict: # Region Readiness Support
    idx = 0
    for i in range(16,22):
        if boxes[i].text.casefold() == 'Lead Proponent Readiness Support:\nRegion Readiness Support:'.casefold(): # Find line based on field name
            idx = i
    text = boxes[idx + 2].text
    score = int(boxes[idx + 1].text.splitlines()[0].strip())
    description = "".join(text.splitlines()).lstrip()
    return {"description":description, "score":score}

def ld_propt_rd_spt(boxes:list) -> dict: # Lead proponent readiness support
    idx = 0
    for i in range(16,22):
        if boxes[i].text.casefold() == 'Lead Proponent Readiness Support:\nRegion Readiness Support:'.casefold(): # Find line based on field name
            idx = i
    
    if len(boxes[idx + 1].text.splitlines())>1:
        text = boxes[idx + 3].text
        score = int(boxes[idx + 1].text.splitlines()[1].strip())
        description = "".join(text.splitlines()).lstrip()
        return {"description":description, "score":score}
    else:
        return {"description":None, "score":None}

def regn_op_cost(boxes:list) -> dict: # Region Operational Cost
    idx = 0
    for i in range(20,27):
        if boxes[i].text.casefold() == 'Lead Proponent Operational Cost:\nRegion Operational Cost:'.casefold(): # Find line based on field name
            idx = i
    text = boxes[idx + 2].text
    score = int(boxes[idx + 1].text.splitlines()[0].strip())
    description = "".join(text.splitlines()).strip()
    return {"description":description, "score":score}

def ld_propt_op_cost(boxes:list)->dict: # Lead proponent operational cost
    idx = 0
    for i in range(20,27):
        if boxes[i].text.casefold() == 'Lead Proponent Operational Cost:\nRegion Operational Cost:'.casefold(): # Find line based on field name
            idx = i
    if len(boxes[idx + 1].text.splitlines())>1:
        text = boxes[idx + 2].text
        score = int(boxes[idx + 1].text.splitlines()[1].strip())
        description = "".join(text.splitlines()[1]).lstrip()
        return {"description":description, "score":score}
    else:
        return {"description":None, "score":None}

def regn_sev_st(boxes:list)->dict: # Region severity statement
    idx = 0
    for i in range(23,30):
        if boxes[i].text.casefold() == 'Lead Proponent Severity Statement:\nRegion Severity Statement:'.casefold(): # Find line based on field name
            idx = i
    text = boxes[idx + 2].text
    score = int(boxes[idx + 1].text.splitlines()[0].strip())
    description = "".join(text.splitlines()).strip()
    return {"description":description, "score":score}

def ld_propt_sev_st(boxes:list)->dict: # Lead proponent severity statement
    idx = 0
    for i in range(23,30):
        if boxes[i].text.casefold() == 'Lead Proponent Severity Statement:\nRegion Severity Statement:'.casefold(): # Find line based on field name
            idx = i
    if len(boxes[idx + 1].text.splitlines())>1:
        text = boxes[idx + 3].text
        score = int(boxes[idx + 1].text.splitlines()[1].strip())
        description = "".join(text.splitlines()).lstrip()
        return {"description":description, "score":score}
    else:
        return {"description":None, "score":None}

def regn_urg_st(boxes:list)->dict: # Region urgency statement
    idx = 0
    for i in range(30,40):
        if boxes[i].text.casefold() == 'Lead Proponent Urgency Statement:\nRegion Urgency Statement:'.casefold(): # Find line based on field name
            idx = i
    text = boxes[idx + 2].text
    score = int(boxes[idx + 1].text.splitlines()[0].strip())
    description = "".join(text.splitlines()).strip()
    return {"description":description, "score":score}

def ld_propt_urg_st(boxes:list)->dict: # Lead proponent urgency statement
    idx = 0
    for i in range(30,40):
        if boxes[i].text.casefold() == 'Lead Proponent Urgency Statement:\nRegion Urgency Statement:'.casefold(): # Find line based on field name
            idx = i
    if len(boxes[idx + 1].text.splitlines())>1:
        text = boxes[idx + 3].text
        score = int(boxes[idx + 1].text.splitlines()[1].strip())
        description = "".join(text.splitlines()).lstrip()
        return {"description":description, "score":score}
    else:
        return {"description":None, "score":None}

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
    # Detect layout: multi-row if we have several boxes at the same y0 (different x0)
    tol = y_cluster_tolerance()
    y_to_boxes: Dict[float, List[TextBox]] = {}
    for b in region_boxes:
        y = round(b.y0 / tol) * tol
        y_to_boxes.setdefault(y, []).append(b)
    same_y_multi = sum(1 for lst in y_to_boxes.values() if len(lst) > 1)
    multi_row_layout = same_y_multi >= 2  # at least 2 rows with multiple cells
    if multi_row_layout:
        return multi_row_parse(region_boxes)
    return single_row_parse(region_boxes)
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
#________________________________________________________________________________________________________________________
files = ["docs/project930.pdf", "docs/project942.pdf"]
cwd = os.getcwd()
file_paths = [os.path.join(cwd, file) for file in files]

proj930 = FormTextExtractor(file_paths[0]).extract_boxes()
proj942 = FormTextExtractor(file_paths[1]).extract_boxes()

proj930_txt = get_pds(proj930)
proj942_txt = get_pds(proj942)

pprint(proj930_txt, width = 100)
pprint(proj942_txt, width = 100)
#________________________________________________________________________________________________________________________








