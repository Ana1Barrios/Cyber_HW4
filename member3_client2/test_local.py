from client2 import Client2

c = Client2()
w = c.get_parameters({})
for r in range(1, 4):
    w, n, _ = c.fit(w, {"server_round": r})   # local training
    c.evaluate(w, {})                          # evaluation on validation split
print("OK: local training works")