from scapy.arch.windows import get_windows_if_list

print("Available interfaces with friendly names:\n")
for iface in get_windows_if_list():
    print(f"👉 Name: {iface['name']}")
    print(f"   GUID: {iface['guid']}")
    print(f"   Description: {iface['description']}")
    print("-" * 60)
