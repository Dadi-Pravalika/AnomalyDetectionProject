from scapy.all import sniff, conf

# Automatically select the default interface
INTERFACE = conf.iface

print(f"🔍 Using interface: {INTERFACE}")
print("Sniffing packets...")

# Try capturing with a timeout in case no packets come
packets = sniff(iface=INTERFACE, count=5, timeout=10)

if len(packets) == 0:
    print("❌ No packets captured. Try increasing timeout or checking network activity.")
else:
    print(f"✅ Captured {len(packets)} packets:")
    packets.summary()
