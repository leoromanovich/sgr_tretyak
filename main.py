import asyncio
import json
import logging

from sgr_agent_core import AgentFactory, GlobalConfig

logging.basicConfig(level=logging.INFO)

NOTE_MD = """
## Метаинформация

![Реквием. Русь уходящая](../images/8331.jpg)

**Автор:** Корин Павел (1892-1967)  
**Название:** Реквием. Русь уходящая  
**Год:** 1935-1959  
**Размер:** 65 x 107,5  
**Материал:** бумага  
**Техника:** гуашь, темпера  
**Инвентарный номер:** РС-2324  
**Происхождение:** Поступило от Всесоюзного художественно-производственного комбината имени Е.В. Вучетича. 1968  

## Описание

Идея создания большой, значимой картины волновала Павла Корина уже в студенческие годы. ...
"""


async def main():
    # 1. Загружаем глобальный конфиг (llm и т.д.)
    config = GlobalConfig.from_yaml("config.yaml")

    # 2. Подгружаем описания агентов из agents.yaml
    config.definitions_from_yaml("agents.yaml")

    # 3. Создаём агента через фабрику — ВАЖНО: именно так, как в доке
    agent = await AgentFactory.create(
        agent_def=config.agents["tretyakov_people_extractor"],
        task=NOTE_MD,
        )

    # 4. Запускаем агента и получаем результат
    result = await agent.execute()   # result — строка (наш JSON)

    print("RAW RESULT:")
    print(result)

    # 5. Парсим JSON
    data = json.loads(result)

    print("\nPARSED PEOPLE:")
    for person in data["people"]:
        print(f"- {person['name']} → {person['role']}")


if __name__ == "__main__":
    asyncio.run(main())