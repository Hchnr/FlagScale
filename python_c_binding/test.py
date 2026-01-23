import simple


result = simple.add(3, 5)
print(f"3 + 5 = {result}")

try:
    simple.add(3, "5")
except TypeError as e:
    print(f"参数错误：{e}")

import pdb

pdb.set_trace()
