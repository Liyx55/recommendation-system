# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['result.py', 'Data\\train.txt', 'Data\\test.txt'],
             pathex=['F:\\大二下\\大数据\\推荐系统编程大作业\\1810756_孙家宜_1813265_李彦欣_1811362_郝旭\\源码'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='result',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True )
