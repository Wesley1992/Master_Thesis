import io
import sys

backup = sys.stdout

sys.stdout = io.StringIO()
print('***** Analysis failed *****')
output = sys.stdout.getvalue()
sys.stdout.close()
sys.stdout = backup

print('1')
print(output)

if 'fails' in output:
    print('failed')

# sys.stdout = buffer = io.StringIO()
# print('***** Analysis failed *****')
# out = buffer.getvalue()
# print('ssss')
# print()
