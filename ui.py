import gradio as gr
from transformers import pipeline, AutoTokenizer
from transformers import AutoTokenizer, BitsAndBytesConfig, pipeline
import transformers
import torch

title="CRG prompt engineering"
desc="Try out multiple models and multiple prompts"
# long_desc="This is a UI for team brainstorming"
model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline("text-generation",model=model,
        model_kwargs={"load_in_4bit": True}, 
        torch_dtype=torch.bfloat16,device_map="auto",)

callback = gr.CSVLogger()


def direct(task, text, prompt, model):

    #out = pipeline(prompt+text,do_sample=True,top_k=10,num_return_sequences=5,eos_token_id=tokenizer.eos_token_id,max_length=100,)
    out = pipeline(prompt+text,do_sample=False,num_return_sequences=5,num_beams=5, num_beam_groups=5,diversity_penalty=10.5,eos_token_id=tokenizer.eos_token_id,max_length=100,)
    print(f"gpu used {torch.cuda.max_memory_allocated(device=None)/1000000000} GB memory")
    #pipe = pipeline("text-generation", model=model,tokenizer=chatbot_tokenizer, num_return_sequences=5, num_beams=5)
    #print("chatbot_tokenizer.encode(chatbot_tokenizer.eos_token = ",chatbot_tokenizer.eos_token)
    #out = pipe(text+chatbot_tokenizer.eos_token)
    text_all = []
    for each in out:
        toappend = each["generated_text"][len(prompt+text):]
        toappend = toappend.split("Visitor")[0]
        text_all.append(toappend+"\n")

    fout=open("continuousLog.txt", "a")
    fout.write("Task:"+task+"\n"+"Model:"+model+"\n"+"Prompt:"+prompt+"\n"+"Context:"+text+"\nResponse1:"+text_all[0]+"\nResponse2:"+text_all[1]+"\nResponse3:"+text_all[2]+"\nResponse4:"+text_all[3]+"\nResponse5:"+text_all[4]+"\n-------------\n")
    fout.close()
    return text_all[0:5]


def predict(task, dialog,prompt, model):
    if(task=="Direct Response Generation"):
        return_text = direct(task, dialog, prompt, model)

    elif(task =="Keyword Generation"):
        return_text = direct(task, dialog, prompt, model)

    elif(task =="Keyword-based Response Generation"):
        return_text = direct(task, dialog, prompt, model)
    return return_text



def load_model(modelchoice, get_resp_btn):
    get_resp_btn = "Wait, model loading!"
    global model
    global tokenizer
    global pipeline
    global llamaLoaded
    global zephyrLoaded
    global phiLoaded
    model = modelchoice
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline("text-generation",model=model,
        model_kwargs={"load_in_4bit": True}, 
        torch_dtype=torch.bfloat16,device_map="auto",trust_remote_code=True)
    get_resp_btn = "Now get response!"
    return get_resp_btn
    

def change_response_button(modelchoice, get_resp_btn):
    get_resp_btn = "Wait, model loading!"
    return get_resp_btn
   


def change_textboxes(taskchoice, context, prompt):
    if(taskchoice=="Keyword Generation"):
        prompt = "The given conversation could have multiple responses. Give out a keyword, one per response, for each possible diverse response."
        context = "Visitor: hey how are you?\nUser: I am good how are you?\nVisitor: I am fine too. What plans for the weekend?\nPossible Response Keywords: "
        return (prompt, context)
    elif(taskchoice=="Keyword-based Response Generation"):
        prompt = "Generate a response as the user, using the given keyword. \n Keyword: sick\n",
        context = "Visitor: hey how are you?\nUser: I am good how are you?\nVisitor: I am fine too. What plans for the weekend?\nUser: "
        return (prompt, context) 
    elif(taskchoice=="Direct Response Generation"):
        prompt = "Generate a response for the given conversation",
        context = "Visitor: hey how are you?\nUser: I am good how are you?\nVisitor: I am fine too. What plans for the weekend?\nUser: "
        return (prompt, context) 


def save(taskchoice, modelchoice, context, prompt , new_title,new_title2,new_title3,new_title4,new_title5):
    fout=open("saved.csv", "a")
    fout.write("Task:"+taskchoice+"\n"+"Model:"+modelchoice+"\n"+"Prompt:"+prompt+"\n"+"Context:"+context+"\nResponse1:"+new_title+"\nResponse2:"+new_title2+"\nResponse3:"+new_title3+"\nResponse4:"+new_title4+"\nResponse5:"+new_title5+"\n-------------\n")
    fout.close()
     


models = ["meta-llama/Llama-2-7b-chat-hf", "HuggingFaceH4/zephyr-7b-beta", "microsoft/phi-2"]
tasks = ["Direct Response Generation", "Keyword Generation", "Keyword-based Response Generation"]

demo = gr.Blocks()
with demo:
    with gr.Row():

        with gr.Column(scale=2):
            gr.Markdown(""" 
            # CRG prompt engineering
            Try out multiple models and multiple prompts 
            """)

            taskchoice = gr.Radio(choices=tasks, label="Task")
            context = gr.Textbox(label="Dialog Context", info="Enter the dialog context for generating response",
                    value="Visitor: hey how are you?\nUser: I am good how are you?\nVisitor: I am fine too. What plans for the weekend?\nUser: ",
                    )
            prompt = gr.Textbox(label="Prompt", info="Prompt for generating response",
                    value="Generate a response as the user, using the given keyword. \nKeyword: sick\n",
                )
            modelchoice = gr.Radio(choices=models, label="Model")

            get_resp_btn = gr.Button("Submit")
            taskchoice.change(fn=change_textboxes, inputs=[taskchoice, context, prompt], outputs=[prompt, context])
            modelchoice.change(fn=change_response_button, inputs=[modelchoice, get_resp_btn], outputs=[get_resp_btn])
            modelchoice.change(fn=load_model, inputs=[modelchoice, get_resp_btn], outputs=[get_resp_btn])

 
        with gr.Column(scale=2):
            new_title = gr.Textbox()
            new_title2 = gr.Textbox()
            new_title3 = gr.Textbox()
            new_title4 = gr.Textbox()
            new_title5 = gr.Textbox()
            get_resp_btn.click(fn=predict, inputs=[taskchoice, context, prompt, modelchoice], outputs=[new_title,new_title2,new_title3,new_title4,new_title5])

            btn = gr.Button("Flag")
            # callback.setup([taskchoice, context, prompt , new_title,new_title2,new_title3,new_title4,new_title5], "flagged_data_points_new")
            # btn.click(lambda *args: callback.flag(args), [taskchoice, context, prompt , new_title,new_title2,new_title3,new_title4,new_title5], None, preprocess=False)
            btn.click(fn=save, inputs=[taskchoice, modelchoice, context, prompt , new_title,new_title2,new_title3,new_title4,new_title5])



demo.launch(share=False,debug=False,server_name="0.0.0.0", server_port=8051)
