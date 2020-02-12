# Tutorial - Remote Entity Linking

## Introduction

The original reason for developing this package at Microsoft is we need a way to Link Entities to a KnowledgeBase without having that KnowledgeBase in memory. 

This tutorial walks through creating the ANN Index for all the Aliases in a KnowledgeBase remotely and exposing the index through a Web Service using FastAPI.

> If you're unfamiliar with FastAPI, you can read more about it here: https://fastapi.tiangolo.com/

> The full code for this tutorial is in [examples/api](https://github.com/microsoft/spacy-ann-linker/tree/master/examples/api)


> This tutorial assumes you've already run the `create_index` command and have a saved model. If you haven't already done that, follow the steps in the [Introduction](../index.md)


The actual webservice service implementation is quite short thanks to FastAPI taking away a lot of the normal boilerplate.

## Load the model

First we need to load our spaCy model from the `create_index` command.

```Python hl_lines="32"
{!./src/api/app.py!}
```

## Define the batch linking route

Once we have our basic API configured and our model loaded, we need a route where we can query the index. Let's add the `/link` route.

```Python hl_lines="40 41"
{!./src/api/app.py!}
```

Now this might seem a bit complicated if you haven't used FastAPI before. However, it's quite simple once you delve into the models. FastAPI leverages Pydantic for serializing and deserializing JSON requests. So at the route level, you pass a response_model and a definition of what the Post Body should look like based on Pydantic models. 

## Models

Let's hop over to our models.py file to get a look at the models our route expects.

```Python hl_lines="18 19 22 23"
{!./src/api/models.py!}
```

If you follow the nested structure of these models you can construct the JSON tree request structure that the API expects as well as the definiton of what it will return.

But we've also included an example_request.json file for a real world example.

## Example Request

```json
{
    "documents": [
        {
            "spans": [
                {
                    "text": "NLP",
                    "start": 0,
                    "end": 3,
                    "label": "SKILL"
                },
                {
                    "text": "researched",
                    "start": 16,
                    "end": 26,
                    "label": "SKILL"
                },
                {
                    "text": "Machine learning",
                    "start": 37,
                    "end": 53,
                    "label": "SKILL"
                }
            ],
            "context": "NLP is a highly researched subset of Machine learning."
        }
    ]
}
```
From this request, you can see that we're passing the Entity Spans extracted by an NER model along with the context in which they were extracted. This is all the input data we need for the `ann_linker` component to be able to identify candidate aliases and disambiguate and alias to a cononical entity id.

Let's hop back over to our main `app.py` and review our `/link` route again.

The logic here is we loop through each document in the request body, make the text into a spaCy Doc object, set the `doc.ents` based on the provided spans and then run the ann_linker pipe on the doc. We then write to the id attribute of each `LinkingSpan` object the value of the `ent.kb_id_` set by the `ann_linker` pipeline component.

```Python hl_lines="51 52 54 55"
{!./src/api/app.py!}
```

## Build and Run

Now that we have an understanding of the code for the Web Service, let's run it using uvicorn.

### 1. Install Requirements

<div class="termy">

```console
$ cd examples/api
$ pip install -r requirements.txt
---> 100%
Successfully installed requirements
```

</div>

### 2. Start the Web Service

<div class="termy">

```console
$ uvicorn app:app --port 8080
Started server process [21052]
Waiting for application startup.
Application startup complete.
Uvicorn running on http://127.0.0.1:8080 (Press CTRL+C to quit)
```

</div>

If you open your browser to http://localhost:8080 now you'll be automatically redirected to the `/docs` route and greeted with the Open API UI for the Web Service

![Open API UI](../img/webservice-openapi.png)

Now if you click on the green highlighted link route, click the button that says "Try it out" and hit Execute, you'll be making a request with the `example_request.json` data and should get a JSON reponse back that looks like:

```json
{
    "documents": [
        {
            "spans": [
                {
                    "text": "NLP",
                    "start": 0,
                    "end": 3,
                    "label": "SKILL",
                    "id": "a3"
                },
                {
                    "text": "researched",
                    "start": 16,
                    "end": 26,
                    "label": "SKILL",
                    "id": "a15"
                },
                {
                    "text": "Machine learning",
                    "start": 37,
                    "end": 53,
                    "label": "SKILL",
                    "id": "a1"
                }
            ],
            "context": "NLP is a highly researched subset of Machine learning."
        }
    ]
}
```

## Call the Web Service

Now that we have an understanding of the remote web service, we need an easy way to call this service from a normal spaCy pipeline. The `RemoteAnnLinker` component handles this interaction.

### Load Extraction Model

First, load a model capable of extracting the Entities in your KnowledgeBase. This could be a trained NER model or a rule based extraction or a combination of both. For simplicity we'll use the spaCy `EntityRuler` component and just add a few terms to it that are close to those in our KnowledgeBase.

```Python hl_lines="4 5 6 7 8"
{!./src/remote_ann_linker.py!}
```

### Create a `remote_ann_linker` pipe

Now create a `remote_ann_linker` pipe using `nlp.create_pipe` and set the `base_url` config value to the batch linking url of your web service. If you're still running the service locally from the last step this should be `http://localhost:8080/link`

```Python hl_lines="10 11 12 13"
{!./src/remote_ann_linker.py!}
```

### Run the pipeline

Now you can call the pipeline the exact same way as you did in when using the local `ann_linker` component and you should get the exact same results.

```Python hl_lines="15 16"
{!./src/remote_ann_linker.py!}
```

## Conclusion

This Web Service is quite simple for the tutorial. It skips over things like a health check url, Docker/Kubernetes based deployment, etc. It's merely meant as a quick guide to illustrate the problem this package was originally designed to solve.
