# Copyright (c) 2025, baddy and contributors
# For license information, please see license.txt

import frappe
from frappe.model.document import Document


class olmocr(Document):
    def on_submit(self):

        frappe.msgprint("Document Submitted")

        file_url = self.file_name

        if not file_url:
            frappe.throw("Please attach a PDF before submitting.")

        file_doc = frappe.get_doc("File", {"file_url": file_url})

        pdf_path = file_doc.get_full_path()

        from olm.utils.auto_pi import process_pdf_and_create_pr

        pr_name = process_pdf_and_create_pr(pdf_path)


