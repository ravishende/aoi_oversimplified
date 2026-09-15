import chainlit as cl

from agent import run_agent


@cl.set_starters
async def set_starters():
    """Show starter cards on the welcome screen."""
    return [
        cl.Starter(
            label="Recommend a model...",
            message="Recommend a model for tabular classification on the Iris dataset at ./data/iris",
            icon="/public/icon-model.svg",
        ),
        cl.Starter(
            label="Search NDP catalog...",
            message="Search for climate datasets in the NDP catalog",
            icon="/public/icon-search.svg",
        ),
    ]

@cl.on_message
async def main(message: cl.Message):
    result = await cl.make_async(run_agent)(message.content)
    await cl.Message(content=result).send()
