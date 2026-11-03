"""NCA morfogenez: planarya benzeri organizma büyüten, onaran ve parçadan yenilenen NCA."""

import ctypes
import os


def _preload_libcuda():
    # NixOS'ta sürücü kütüphanesi standart yolda değil; torch CUDA'yı başlatmadan önce
    # libcuda'yı soname ile yükleriz ki torch'un dlopen çağrısı onu bulsun.
    for path in ("/run/opengl-driver/lib/libcuda.so.1",):
        if os.path.exists(path):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
            return


_preload_libcuda()
