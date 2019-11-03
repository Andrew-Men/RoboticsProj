import sys
import pydoc
import cv2

def output_help_to_file(filepath, request):
    f = open(filepath, 'w')
    sys.stdout = f
    pydoc.help(request)
    f.close()
    sys.stdout = sys.__stdout__
    return

output_help_to_file(r'ArUcoHelp','cv2.aruco')