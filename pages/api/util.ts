import { OpenAIChatChat } from "langchain/llms";
import { ChatChatVectorDBQAChain } from "langchain/chains";
import { HNSWLib } from "langchain/vectorstores";

export const makeChain = (
  vectorstore: HNSWLib,
  onTokenStream?: (token: string) => void
) => {
  return ChatChatVectorDBQAChain.fromModel(
    new OpenAIChatChat({
      role: "assistant",
      streaming: Boolean(onTokenStream),
      callbackManager: {
        handleNewToken: onTokenStream,
      },
    }),
    vectorstore,
    [
      {
        role: "system",
        text: `You are an AI assistant for the open source library LangChain. The documentation is located at https://langchain.readthedocs.io.
You are given the following extracted parts of a long document and a question. Provide a conversational answer with a hyperlink to the documentation.
You should only use hyperlinks that are explicitly listed as a source in the context. Do NOT make up a hyperlink that is not listed.
If the question includes a request for code, provide a code block directly from the documentation.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about LangChain, politely inform them that you are tuned to only answer questions about LangChain.
Always format your answer in Markdown.`,
      },
    ]
  );
};
