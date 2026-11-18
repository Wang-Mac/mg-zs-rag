from langchain_chroma import Chroma
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_openai import ChatOpenAI

from mg_zs_rag.config import (
    CHROMA_DIR,
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_EMBEDDING_MODEL,
    NVIDIA_LLM_MODEL,
    RETRIEVER_K,
    check_chat_config,
)

# 1. store conversation history
store = {}

# 根据id获取对应的聊天记录
def get_session_history(session_id:str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 2. Format retrieved documents
def format_docs(docs):
    return "\n\n".join(
        f"[source: {doc.metadata.get('source','MG ZS Manual')}]\n"
        f"{doc.page_content}"
        for doc in docs
    )

# 3. build rag chain

def build_rag_chain():

    check_chat_config()

    # 两个模型
    llm = ChatOpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=NVIDIA_API_KEY, # type: ignore
        model=NVIDIA_LLM_MODEL,
        temperature=0,
        model_kwargs={
            'max_tokens':4096
        }
    )

    embeddings = NVIDIAEmbeddings(
        base_url=NVIDIA_BASE_URL,
        api_key=NVIDIA_API_KEY,
        model=NVIDIA_EMBEDDING_MODEL,
        truncate="END"
    )


# 创建检索器

    # 连接chroma数据库
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name='mg_zs_manual'
    )

    #检查是否数据库中含有数据
    if not vectorstore.get(limit=1)['ids']:
        raise ValueError(
            "Chroma database is empty. Please rebuild the knowledge base."           
        )

    #检索器
    retriever = vectorstore.as_retriever(
        search_type = 'similarity',
        search_kwargs = {'k': RETRIEVER_K}
    )

# 创建rag链
    
    #创建 重写promot
    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Given the conversation history and the latest user
                question, rewrite the question into a standalone question.

                Do not answer the question.

                If the question is already independent,
                return it unchanged.
                """
            ),
            MessagesPlaceholder(
                variable_name='chat_history'
            ),
            ('human', '{input}')
        ]
    )

    #封装 重写prompt chain
    contextualize_chain = (
        contextualize_prompt
        | llm
        | StrOutputParser()
    )

    #构建 回答prompt
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an MG ZS automotive manual assistant.

                Answer the user's question using only the retrieved
                information from the official vehicle manual.

                Requirements:

                1. Do not invent warning meanings or repair instructions.

                2. If the retrieved context does not contain the answer,
                    clearly state that the manual information is insufficient.

                3. Explain what the warning means and what action
                    the driver should take, when the manual provides
                    that information.

                4. Never claim that it is safe to continue driving
                    unless the manual explicitly supports this.

                5. If an immediate safety risk is apparent,
                    advise the driver to stop in a safe place
                    and seek qualified assistance.

                6. Respond in the same language as the user.

                7. Keep the answer concise and easy to understand.

                8. Only state facts explicitly supported by the retrieved manual context.

                9. Never invent fault causes, diagnostic procedures, thresholds, or repair steps.

                10. Do not present general automotive knowledge as official manual instructions.

                11. If the manual does not specify something, explicitly say: "当前检索到的手册内容没有说明这一点。"

                12. Clearly distinguish manual instructions from any additional safety advice.

                Retrieved manual context:

                {context}
                """
            ),
            ('human','{question}')
        ]
    )

    
    # 构建检索函数
    def retriever_context(inputs):
        question = inputs['input']
        chat_history = inputs['chat_history']

        #通过重写prompt实现将chat_history包装进prompt
        if chat_history:
            question = contextualize_chain.invoke(
                {
                    'input': inputs['input'],
                    'chat_history': chat_history
                }
            )

        docs = retriever.invoke(question)

        return {
            'question': question,
            'context': format_docs(docs)
        }

    # 构建rag chain
    rag_chain = (
        RunnableLambda(retriever_context)
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    #实现多轮问答
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history'
    )

    return conversational_rag_chain



def main():
    rag_chain = build_rag_chain()

    session_id = 'mg-zs-session-001'

    print("=" * 50)
    print("MG ZS Manual Assistant")
    print("Type 'exit' to quit.")
    print("=" * 50)

    while True:
        query = input('\nYou: ').strip()

        if query.lower() in {'exit','quit'}:
            print('Goodbye')
            break
        if not query:
            continue

        try:
            response = rag_chain.invoke(
                {'input': query},
                config = {
                    'configurable': {
                        'session_id': session_id
                    }
                }
            )

            print('\nAssistant:')
            print(response)
        

        except Exception as e: # noqa: BLE001
            print(f'\nerror: {e}')

if __name__ == '__main__':
    main()

    