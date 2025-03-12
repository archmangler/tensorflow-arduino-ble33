import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

filename = "sidelift.csv"

df = pd.read_csv("./data/" + filename)

index = range(1, len(df['aX']) + 1)

plt.rcParams["figure.figsize"] = (20,10)

plt.plot(index, df['aX'], 'g.', label='x', linestyle='solid', marker=',')
plt.plot(index, df['aY'], 'b.', label='y', linestyle='solid', marker=',')
plt.plot(index, df['aZ'], 'r.', label='z', linestyle='solid', marker=',')
plt.title("Sidelift Curl Acceleration")
plt.xlabel("Sidelift Curl Sample #")
plt.ylabel("Sidelift Curl Acceleration (G)")
plt.legend()
plt.show()

plt.plot(index, df['gX'], 'g.', label='x', linestyle='solid', marker=',')
plt.plot(index, df['gY'], 'b.', label='y', linestyle='solid', marker=',')
plt.plot(index, df['gZ'], 'r.', label='z', linestyle='solid', marker=',')
plt.title("Sidelift Curl Gyroscope")
plt.xlabel("Sidelift Curl Sample #")
plt.ylabel("Sidelift Curl Gyroscope (deg/sec)")
plt.legend()
plt.show()

filename = "rotcurl.csv"

df = pd.read_csv("./data/" + filename)

index = range(1, len(df['aX']) + 1)

plt.rcParams["figure.figsize"] = (20,10)

plt.plot(index, df['aX'], 'g.', label='x', linestyle='solid', marker=',')
plt.plot(index, df['aY'], 'b.', label='y', linestyle='solid', marker=',')
plt.plot(index, df['aZ'], 'r.', label='z', linestyle='solid', marker=',')
plt.title("Rotating Curl Acceleration")
plt.xlabel("Rotating Curl Sample #")
plt.ylabel("Rotating Curl Acceleration (G)")
plt.legend()
plt.show()

plt.plot(index, df['gX'], 'g.', label='x', linestyle='solid', marker=',')
plt.plot(index, df['gY'], 'b.', label='y', linestyle='solid', marker=',')
plt.plot(index, df['gZ'], 'r.', label='z', linestyle='solid', marker=',')
plt.title("Rotating Curl Gyroscope")
plt.xlabel("Rotating Curl Sample #")
plt.ylabel("Rotating Curl Gyroscope (deg/sec)")
plt.legend()
plt.show()

filename = "curl.csv"

df = pd.read_csv("./data/" + filename)

index = range(1, len(df['aX']) + 1)

plt.rcParams["figure.figsize"] = (20,10)

plt.plot(index, df['aX'], 'g.', label='x', linestyle='solid', marker=',')
plt.plot(index, df['aY'], 'b.', label='y', linestyle='solid', marker=',')
plt.plot(index, df['aZ'], 'r.', label='z', linestyle='solid', marker=',')
plt.title("Curl Acceleration")
plt.xlabel("Curl Sample #")
plt.ylabel("Curl Acceleration (G)")
plt.legend()
plt.show()

plt.plot(index, df['gX'], 'g.', label='x', linestyle='solid', marker=',')
plt.plot(index, df['gY'], 'b.', label='y', linestyle='solid', marker=',')
plt.plot(index, df['gZ'], 'r.', label='z', linestyle='solid', marker=',')
plt.title("Curl Gyroscope")
plt.xlabel("Curl Sample #")
plt.ylabel("Curl Gyroscope (deg/sec)")
plt.legend()
plt.show()

filename = "punch.csv"

df = pd.read_csv("./data/" + filename)

index = range(1, len(df['aX']) + 1)

plt.rcParams["figure.figsize"] = (20,10)

plt.plot(index, df['aX'], 'g.', label='x', linestyle='solid', marker=',')
plt.plot(index, df['aY'], 'b.', label='y', linestyle='solid', marker=',')
plt.plot(index, df['aZ'], 'r.', label='z', linestyle='solid', marker=',')
plt.title("Acceleration")
plt.xlabel("Punch Sample #")
plt.ylabel("Punch Acceleration (G)")
plt.legend()
plt.show()

plt.plot(index, df['gX'], 'g.', label='x', linestyle='solid', marker=',')
plt.plot(index, df['gY'], 'b.', label='y', linestyle='solid', marker=',')
plt.plot(index, df['gZ'], 'r.', label='z', linestyle='solid', marker=',')
plt.title("Gyroscope")
plt.xlabel("Punch Sample #")
plt.ylabel("Punch Gyroscope (deg/sec)")
plt.legend()
plt.show()

#

filename = "flex.csv"

df = pd.read_csv("./data/" + filename)

index = range(1, len(df['aX']) + 1)

plt.rcParams["figure.figsize"] = (20,10)

plt.plot(index, df['aX'], 'g.', label='x', linestyle='solid', marker=',')
plt.plot(index, df['aY'], 'b.', label='y', linestyle='solid', marker=',')
plt.plot(index, df['aZ'], 'r.', label='z', linestyle='solid', marker=',')
plt.title("Flex Acceleration")
plt.xlabel("Flex Sample #")
plt.ylabel("Flex Acceleration (G)")
plt.legend()
plt.show()

plt.plot(index, df['gX'], 'g.', label='x', linestyle='solid', marker=',')
plt.plot(index, df['gY'], 'b.', label='y', linestyle='solid', marker=',')
plt.plot(index, df['gZ'], 'r.', label='z', linestyle='solid', marker=',')
plt.title("Flex Gyroscope")
plt.xlabel("Flex Sample #")
plt.ylabel("Flex Gyroscope (deg/sec)")
plt.legend()
plt.show()

