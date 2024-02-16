import subprocess

dt = {'c0017': [72, 768, 5],
      'c0018': [72, 384, 5],
      'c0019': [72, 384, 5],
      'c0020': [72, 384, 5],
      'c0021': [72, 384, 5],
      'c0022': [72, 1536, 2],
      'c0071': [112, 1024, 4],
      'c0072': [112, 1024, 4],
      'c0073': [112, 1024, 4],
      'c0074': [112, 1024, 4],
      'c0076': [112, 1024, 4],
     }


'''check living nodes'''
live_node = []
out = subprocess.check_output(["sinfo", "-p", "aida"])
out = out.decode()
lines = out.rstrip().split('\n')
for line in lines:
    items = line.split()
    if items[-2] not in ['mix', 'idle']:
        continue
    if '[' not in items[-1]:
        live_node.append(items[-1])
        continue
    items = items[-1][2:-1].split(',')
    for item in items:
        if '-' in item:
            a, b = item.split('-')
            for x in range(int(a), int(b) + 1):
                live_node.append(f"c00{x}")
        else:
            live_node.append(f"c{item}")
print(live_node)

for key in list(dt.keys()):
    if key not in live_node:
        del dt[key]


out = subprocess.check_output(["squeue", "-p", "aida", "-o", "'%.6i %.9P %.8j %.8u %.2t %.10M %.10l %.6D %.4C %.7m %.20b %.30R'"])
out = out.decode()
lines = out.rstrip().split('\n')

for line in lines:
    items = line[1:-1].split()
    if items[4] != 'R':
        continue

    node = items[-1]
    if node not in dt:
        continue
    dt[node][0] -= int(items[-4])
    mem = items[-3]
    assert mem[-1] in ['T', 'G', 'M']
    if mem[-1] == 'T':
        dt[node][1] -= int(mem[:-1]) * 1024
    elif mem[-1] == 'G':
        dt[node][1] -= int(mem[:-1])
    elif mem[-1] == 'M':
        dt[node][1] -= int(mem[:-1]) / 1024
    if 'gpu' in items[-2]:
        dt[node][2] -= int(items[-2].split(':')[-1])

'''
for key in list(dt.keys()):
    if dt[key][-1] == 0:
        del dt[key]
'''
    
print(dt)

