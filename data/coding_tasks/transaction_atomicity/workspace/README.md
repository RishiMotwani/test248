# writeapp

Write-layer facade over a fake upstream client. `Writer.submit(client,
payload)` performs the two-step write through `write_record` and `write_audit`
in order. Upstream failures surface as `UpstreamError`.