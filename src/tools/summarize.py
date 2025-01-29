import json
import logging
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from typing import Any, List, Tuple
from langchain_openai import OpenAI
from pydantic import BaseModel, Field
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.output_parsers.json import parse_json_markdown
from tenacity import retry, stop_after_attempt, wait_fixed
from src.tools.lang_detector import detect_language
from src.utils.preprocessor import preprocess_text
from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models import BaseLanguageModel



default_combine_prompt = """
    Write a concise summary in French of the following text delimited by triple backquotes.
    Return your response in bullet points which covers the key points of the text.
    ```{text}```
    BULLET POINT SUMMARY:
    """

default_map_prompt = """
    Write a concise summary of the following:
    "{text}"
    CONCISE SUMMARY:
    """

def build_prompt(template: str, lang: str) -> str:
    """Helper to format prompts."""
    return template.replace("{lang}",lang)

async def summarize_text(text:str, bllm:BaseLanguageModel, combine_prompt:str = default_combine_prompt, map_prompt=default_map_prompt):
    """This function takes as text input, combine prompt and returns its summary"""
    try:
        #print("model="+llm.model_config.)
        docs = [Document(page_content=text)]
        _map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
        _combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
        summary_chain = load_summarize_chain(llm=bllm,
                                        chain_type='map_reduce',
                                        map_prompt=_map_prompt_template,
                                        combine_prompt=_combine_prompt_template,
    #                                      verbose=True
                                        )
        output = await summary_chain.arun(docs)
        return output
    except Exception as e:
        print(e)
        raise Exception("Error in summarize_text "+str(e) )
    
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def summarize_text_with_retry(*args, **kwargs):
    """Wrapper to retry text summarization in case of transient errors."""
    return await summarize_text(*args, **kwargs)


async def summarize_Documents(docs:List[Document], llm:BaseChatModel, custom_prompt=default_combine_prompt):
    print("summarize_Documents")
    combine_prompt_template = PromptTemplate(template=custom_prompt, input_variables=["text"])
    print("combine_prompt_template="+str(combine_prompt_template))
    summary_chain = load_summarize_chain(llm=llm,
                                    chain_type='map_reduce',
                                    combine_prompt=combine_prompt_template,
                                    )
    summary =  summary_chain.invoke(docs)
    return summary

async def summarize_articles_from_urls_returns_sorted_by_rank(urls:List[str],instructions_for_ranking:str,llm:BaseChatModel)->List[Any]:
    scored_article_results=[]
    executor = BSQueryExecutor()

    queries={
        "https://www.ecommercemag.fr":[{
        "action":"find_all",
        "tag":"div",
        "attributes":{"class":"qiota_reserve"},
        "extract":["text"]
        }],
        "https://www.republik-retail.fr":[{
        "action":"find",
        "tag":"article",
        "extract":["text"]
        }],
        "https://www.journaldunet.com/retail":[{
        "action":"find",
        "tag":"div",
        "attributes":{"id":"jArticleInside"},
        "extract":["text"]
        }]
    }
        
    for url in urls:
        print("summarizing "+url)
        base_url = get_base_url(url)
        print("base_url="+base_url)
        query = queries.get(base_url)
        if query is None:
            print("no query for this base url")
            query=[{
        "action":"find_all",
        "tag":"p",
        "extract":["text"]
        }]

        docs = AsyncHtmlLoader(url).load()
        
        if(len(docs)==0):
            continue
        content = docs[0]
        #print(content)
        content = executor.query(url, content.page_content, query) 
        if(len(content)==0):
            continue
        html_text=""
        for c in content:
            html_text+=c["text"]+"\n"

        docs[0].page_content=html_text

        prompt_for_summarization = instructions_for_ranking + """```{text}```. 
        Important: return only a markdown formatted json with three properties: 'summary', 'relevance' and 'insightfulness'. 'relevance' and 'insightfulness' are integers ranking from 1 to 5.         
        """
        try:
            text = await summarize_text(text=html_text,combine_prompt= prompt_for_summarization, bllm=llm)
            print(text)
            print("formatting")
            try:
                formatted=json.loads(text)    
            except:
                formatted = parse_json_markdown(text)

            
            print(formatted)
            relevance= int(formatted.get("relevance"))
            insightfulness= int(formatted.get("insightfulness"))
            result={
                "url":url,
                "text":formatted.get("summary"),
                "rank":relevance + insightfulness,
                "relevance":relevance,
                "insightfulness":insightfulness,
            }
            print(result)
            scored_article_results.append(result)
        except Exception as error:
            # handle the exception
            print("An exception occurred:", error)
            print("ERROR***********************")
    print(scored_article_results)
    return scored_article_results

class TextSummarizer:
    default_combine_prompt = """
        Write a concise summary in {lang} of the following text delimited by triple backquotes.
        Return your response in bullet points which covers the key points of the text.
        ```{text}```
        BULLET POINT SUMMARY:
        """
    default_map_prompt = """
        Write a concise summary in {lang} of the following:
        "{text}"
        CONCISE SUMMARY:
        """
    llm:BaseChatModel
    def __init__(self, llm:BaseChatModel=None):
        if llm is None:
            self.llm =  OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.7, max_tokens=150)


    async def summarize(self, text: str, lang: str = None, custom_combine_prompt:str=None, custom_map_prompt:str=None) -> Tuple[str,str]:
        preprocessed_text = preprocess_text(text)
        if lang is None:
            lang = detect_language(preprocessed_text)
        combine_prompt = custom_combine_prompt if custom_combine_prompt is not None else self.default_combine_prompt
        map_prompt = custom_map_prompt if custom_map_prompt is not None else self.default_map_prompt
        combine_prompt = build_prompt(combine_prompt, lang)
        map_prompt = build_prompt(map_prompt, lang)
        summarized_text = await summarize_text_with_retry(
            text=preprocessed_text,
            bllm=self.llm,
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
        )
        return (lang, summarized_text)
    
    def summarize_texts(self,texts: List[str]) -> str:
        """Summarize a list of texts using OpenAI."""
        logging.info(f"Summarizing {len(texts)} texts")
        prompt = ChatPromptTemplate.from_template(
            "Summarize the following text concisely:\n\n{text}"
        )
        chain = prompt | self.llm
        input_data = {"text": texts}
        return chain.invoke(input_data)




import asyncio

if __name__ == "__main__":
    # Example usage
    text = "This is a long text that needs to be summarized. It contains multiple sentences and paragraphs. The goal is to create a concise summary that captures the essence of the text in a few sentences."

    summarizer = TextSummarizer()

    async def main():
        lang, summarized_text = await summarizer.summarize(text,"fr")
        print(summarized_text)

    asyncio.run(main())  # Properly run the async function

