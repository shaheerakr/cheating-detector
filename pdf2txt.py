#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 19:48:36 2020

@author: shaheer
"""

try:
    from xml.etree.cElementTree import XML
except ImportError:
    from xml.etree.ElementTree import XML
import zipfile

WORD_NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
PARA = WORD_NAMESPACE + 'p'
TEXT = WORD_NAMESPACE + 't'

def get_docx_text(doc):
    document = zipfile.ZipFile(doc)
    xml_content = document.read('word/document.xml')
    document.close()
    tree = XML(xml_content)

    paragraphs = []
    for paragraph in tree.getiterator(PARA):
        texts = [node.text
                 for node in paragraph.getiterator(TEXT)
                 if node.text]
        if texts:
            paragraphs.append(''.join(texts))

    return '\n\n'.join(paragraphs)

import base64
import io
with open("Muhammad Shaheer Akram.docx","rb") as doc_file:
    doc_encoded = base64.b64encode(doc_file.read())

import requests

docs = []
docs.append(doc_encoded)

res = requests.post('http://ab756eb9.ngrok.io/word-text/123', 
                    json = {"docs": docs})
if res.ok:
    result = res.json()
    print(res.json())
else:
    print("somthing went wrong")


doc = base64.b64decode(doc_encoded)
doc = io.BytesIO(doc)

path = "Muhammad Shaheer Akram.docx";
print(get_docx_text(doc))