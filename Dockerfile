FROM python:3.8

ENV MICRO_SERVICE=/home/app/webapp
# set work directory
RUN mkdir -p $MICRO_SERVICE
# where your code lives
WORKDIR $MICRO_SERVICE

# copy requirements.txt
COPY requirements.txt $MICRO_SERVICE/requirements.txt
# install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# copy project
COPY . $MICRO_SERVICE
# start streamlit app
CMD ["sh", "-c", "streamlit run --server.port $PORT app.py"]