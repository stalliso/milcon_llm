# Code with all functions necessary to parse our Project Data Sheets

import fitz  # pymupdf
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import os
from pprint import pprint

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
            "lead_proponent_urgency_statement":ld_propt_urg_st(boxes)
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









