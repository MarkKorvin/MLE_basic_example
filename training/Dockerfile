FROM python:2.7
WORKDIR /training
COPY . .
RUN pip install -r requirements.txt
RUN npm install
CMD [ "npm", "start" ]