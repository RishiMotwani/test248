# configapp

Configuration codec. `ConfigCodec` parses `key=value` text with
`parse(text)`, serializes a mapping with `serialize(mapping)`, and
`load(text, spec)` merges the spec's defaults into the parsed config.