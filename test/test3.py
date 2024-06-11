import asyncio
from datetime import datetime


async def sleep():
    await asyncio.sleep(3)


async def test(ident):
    print(ident, datetime.now())
    await sleep()
    print(ident, datetime.now())


async def test2():
    ident_list = ['A', 'B', 'C']
    start = datetime.now()
    await asyncio.gather(
        *[
            test(ident) for ident in ident_list
        ]
    )
    print((datetime.now() - start).total_seconds())

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(test2())