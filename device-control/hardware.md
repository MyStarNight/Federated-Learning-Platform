# Hardware Information

# IP and Mac

## Position and Location

| Ubuntu-Laptop       | IP           | Mac               |
| ------------------- | ------------ | ----------------- |
| hao-virtual-machine | 192.168.3.17 | 00:0C:29:F9:25:04 |

| 三教pic-Raspi | IP           | Mac               |
| ------------- | ------------ | ----------------- |
| 1             | 192.168.3.12 | DC:A6:32:6B:9E:0A |
| 2             | 192.168.3.20 | DC:A6:32:6B:A2:9C |
| 3             | 192.168.3.8  | DC:A6:32:6B:A0:DA |
| 4             | 192.168.3.3  | DC:A6:32:6B:A2:4B |

| 流汗pic-Raspi | IP           | Mac               |
| ------------- | ------------ | ----------------- |
| 1             | 192.168.3.7  | DC:A6:32:6B:A0:FE |
| 2             | 192.168.3.11 | DC:A6:32:6B:A1:23 |
| 3             | 192.168.3.2  | DC:A6:32:3A:3F:29 |
| 4             | 192.168.3.13 | DC:A6:32:6B:A1:0A |

| Single-Raspi | IP           | Mac               |
| ------------ | ------------ | ----------------- |
| Unit-01      | 192.138.3.4  | E4:5F:01:03:EB:E7 |
| Unit-02      | 192.168.3.10 | DC:A6:32:AD:CC:B3 |

| Jetson Nano 1 + 3 | IP           | Mac               |
| ----------------- | ------------ | ----------------- |
| Unit-01           | 192.168.3.15 | 00:04:4B:EB:A0:50 |
| 1                 | 192.168.3.22 | 00:04:4B:EB:A0:20 |
| 2                 | \            | \                 |
| 3                 | 192.168.3.23 | 00:04:4B:EB:9D:25 |

| Jetson Nano 4 | IP           | Mac               |
| ------------- | ------------ | ----------------- |
| 1             | 192.168.3.5  | 00:04:4B:E5:2F:5A |
| 2             | 192.168.3.9  | 00:04:4B:E5:2F:4F |
| 3             | 192.168.3.6  | 00:04:4B:E6:2B:16 |
| 4             | 192.168.3.16 | 00:04:4B:E6:2F:54 |

## Sequence

| Raspi        | Jetson Nano  |
| ------------ | ------------ |
| 192.168.3.2  | 192.168.3.5  |
| 192.168.3.3  | 192.168.3.6  |
| 192.168.3.4  | 192.168.3.9  |
| 192.168.3.7  | 192.168.3.15 |
| 192.168.3.8  | 192.168.3.16 |
| 192.168.3.10 | 192.168.3.22 |
| 192.168.3.11 | 192.168.3.23 |
| 192.168.3.12 |              |
| 192.168.3.13 |              |
| 192.168.3.20 |              |



# client devices in python dict format

```python
    raspberries = [
        {"host": "192.168.3.2", "hook": hook, "verbose": args.verbose},
        {"host": "192.168.3.3", "hook": hook, "verbose": args.verbose},
        {"host": "192.168.3.4", "hook": hook, "verbose": args.verbose},
        {"host": "192.168.3.7", "hook": hook, "verbose": args.verbose},
        {"host": "192.168.3.8", "hook": hook, "verbose": args.verbose},
        {"host": "192.168.3.10", "hook": hook, "verbose": args.verbose},
        {"host": "192.168.3.11", "hook": hook, "verbose": args.verbose},
        {"host": "192.168.3.12", "hook": hook, "verbose": args.verbose},
        {"host": "192.168.3.13", "hook": hook, "verbose": args.verbose},
        {"host": "192.168.3.20", "hook": hook, "verbose": args.verbose},
    ]

    jetson_nano = [
        {"host": "192.168.3.5", "hook": hook, "verbose": args.verbose},
        {"host": "192.168.3.6", "hook": hook, "verbose": args.verbose},
        {"host": "192.168.3.9", "hook": hook, "verbose": args.verbose},
        {"host": "192.168.3.15", "hook": hook, "verbose": args.verbose},
        {"host": "192.168.3.16", "hook": hook, "verbose": args.verbose},
        {"host": "192.168.3.22", "hook": hook, "verbose": args.verbose},
        {"host": "192.168.3.23", "hook": hook, "verbose": args.verbose},
    ]
```

