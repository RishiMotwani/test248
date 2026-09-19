from invoiceapp.adapter import normalize_invoice


def test_maps_unknown_line_item_codes():
    row = {"invoice_id": 9, "line_items": [
        {"id": 1, "code": "LINE-07"},
        {"id": 2, "code": "VENDOR-X"},
    ]}
    out = normalize_invoice(row, {})
    assert [i["code"] for i in out["line_items"]] == ["MISCELLANEOUS", "MISCELLANEOUS"]


def test_known_codes_survive():
    row = {"invoice_id": 3, "line_items": [{"id": 1, "code": "A"}]}
    out = normalize_invoice(row, {"A": "Known"})
    assert out["line_items"][0]["code"] == "A"
