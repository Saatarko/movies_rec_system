# task_registry.py

TASK_HANDLERS = {}


def task(name):
    """
    Decorator function
    Attributes:
    name - task name
    """

    def wrapper(func):
        TASK_HANDLERS[name] = func
        return func

    return wrapper


def main(task_names):
    """
    Task handling function for DVC from TASK_HANDLERS list
    Attributes:
        task_names - task name
    """
    for task_name in task_names:
        if task_name not in TASK_HANDLERS:
            raise ValueError(
                f"Неизвестная задача: {task_name}\nДоступные задачи: {list(TASK_HANDLERS.keys())}"
            )
        print(f"\n=== Выполняется задача: {task_name} ===")
        TASK_HANDLERS[task_name]()
