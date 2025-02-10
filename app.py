from flask import Flask, render_template,request
import pickle
app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")
@app.route("/predict",methods=["POST","Get"])
def predict():
    if request.method=="POST":
        msg=request.form.get("message")
        print(msg)
        ob=TextToNum(msg)
        ob.cleaner()
        ob.token()
        ob.removeStop()
        st=ob.stemme()
        stem_vector=" ".join(st)

        with open("vectorizer.pickle","rb") as vc:
            vectorizer=pickle.load(vc)
        vcdata=vectorizer.transform([stem_vector]).toarray()
        print(vcdata)
        
        with open("model.pickle","rb") as mc:
            model=pickle.load(mc)

        pred=model.predict(vcdata)
        print(pred)


    else:
        return render_template("predict.html")

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5151)