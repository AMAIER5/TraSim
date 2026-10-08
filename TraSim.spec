# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = [('gui', 'gui')]
binaries = []
hiddenimports = []
tmp_ret = collect_all('streamlit')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# gui/app.py is bundled as data, so PyInstaller's static analysis never
# follows its imports.  Bundle the local packages it needs explicitly.
local_packages = [
    'analysis',
    'core',
    'mechanics',
    'mechanism_io',
    'model',
    'optimization',
    'simulation',
    'validation',
]
hiddenimports += local_packages
for package in local_packages:
    submodules = collect_submodules(package)
    hiddenimports += submodules


a = Analysis(
    ['run_trasim.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='TraSim',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
