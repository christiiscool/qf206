from pathlib import Path

replacements = {
    Path('src/features.py'): [('.resample("M")', '.resample("ME")')],
    Path('src/factor_timing_model.py'): [('.resample("M")', '.resample("ME")')],
    Path('src/config.py'): [
        ('calendar: str = "BM"', 'calendar: str = "BME"'),
        ('rebalance_freq: str = "M"', 'rebalance_freq: str = "ME"'),
        ('options_snapshot_freq: str = "M"', 'options_snapshot_freq: str = "ME"'),
    ],
}

for path, pairs in replacements.items():
    text = path.read_text(encoding='utf-8')
    original = text
    for old, new in pairs:
        text = text.replace(old, new)
    if text != original:
        path.write_text(text, encoding='utf-8')
        print(f'Updated {path}')
    else:
        print(f'No changes needed for {path}')
