#github code

import torch
import base64
import yaml
import requests

from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import frappe

from olmocr.data.renderpdf import render_pdf_to_base64png


SEMANTIC_URL = "http://127.0.0.1:8500/semantic_search"   
MATCH_THRESHOLD = 0.65  


#. GITHUB CODE 

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "allenai/olmOCR-2-7B-1025",
    torch_dtype=torch.bfloat16
).eval()

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def run_ocr_to_dict(pdf_path: str) -> dict:

    image_base64 = render_pdf_to_base64png(pdf_path, 1, target_longest_image_dim=1288)

    prompt = """
You are an OCR extraction system.

Attached is one page of a purchase invoice. 
Extract ONLY structured data as valid YAML with this exact schema:

---
supplier: <supplier name>
company: <company name>
items:
  - name: <item name>
    quantity: <integer quantity>
    amount: <line total amount exactly as printed>
total_amount: <overall total amount>
---

Rules:
- Return ONLY YAML, nothing else.
- quantity must be an integer (255, 255.0, 255,00 → 255)
- amount must be extracted exactly as printed (1862,50 stays 1862,50)
- Do NOT output markdown, html, or tables.
    """

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                },
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

    inputs = processor(
        text=[text],
        images=[main_image],
        padding=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output = model.generate(
        **inputs,
        temperature=0.1,
        max_new_tokens=3500,
        do_sample=True
    )

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_len:]
    text_output = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


    #github code end

    yaml_output = text_output[0]
    return yaml.safe_load(yaml_output)


def normalize_amount_to_float(text: str) -> float:
    text = text.replace(" ", "")
    return float(text.replace(".", "").replace(",", "."))


def semantic_match_item_name(raw_name: str) -> str:
    resp = requests.post(SEMANTIC_URL, json={"query": raw_name})
    resp.raise_for_status()

    data = resp.json()
    results = data.get("results", [])

    if not results:
        frappe.throw(f"No semantic match found for: {raw_name}")

    best = results[0]
    score = best["score"]
    best_name = best["name"]

    if score < MATCH_THRESHOLD:
        frappe.throw(
            f"Weak match for \"{raw_name}\".\n"
            f"Closest: {best_name} (score {score:.2f})"
        )

    return best_name

def create_purchase_receipt_from_dict(data: dict) -> str:

    supplier_name = data["supplier"]

    supplier_id = frappe.db.get_value("Supplier", {"supplier_name": supplier_name}, "name")

    if not supplier_id:
        frappe.throw(f"Supplier {supplier_name} not found in ERPNext")

    supplier = supplier_id
    company = data["company"]
    items = data["items"]

    items_table = []
    matched_count = 0  

    for row in items:
        raw_name = row["name"]
        qty = int(row["quantity"])
        line_total = normalize_amount_to_float(str(row["amount"]))

        try:
            matched_name = semantic_match_item_name(raw_name)

            item_code = frappe.db.get_value("Item", {"item_name": matched_name}, "name")

            if not item_code:
                continue  

            rate = line_total / qty if qty else 0

            items_table.append({
                "item_code": item_code,
                "qty": qty,
                "rate": rate
            })

            matched_count += 1  

        except Exception:
             continue

    if matched_count == 0:
        frappe.throw("None of the items could be matched. No Purchase Report created.")

    pr = frappe.get_doc({
        "doctype": "Purchase Receipt",
        "supplier": supplier,
        "company": company,
        "posting_date": frappe.utils.nowdate(),
        "items": items_table,
        "set_warehouse":"Stores - ANEL",
    })

    pr.insert()
    pr.submit()
    return pr.name

def process_pdf_and_create_pr(pdf_path: str) -> str:
    data = run_ocr_to_dict(pdf_path)
    pr_name = create_purchase_receipt_from_dict(data)
    frappe.msgprint(f" Purchase Receipt created: {pr_name}")
    return pr_name
