from dm_control import suite

print("before")
env = suite.load("cartpole", "balance")
print("after")