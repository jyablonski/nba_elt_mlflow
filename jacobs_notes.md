### ML FLOW
# Basics
run `src/main.py` to run the model
  
run `mlflow ui` and go to `http://localhost:5000/#/` to view historical runs.  it's just a flask server.

`mlflow run sklearn_elasticnet_wine -P alpha=0.5`
`mlflow run https://github.com/mlflow/mlflow-example.git -P alpha=5.0`


have the tonights ml game have the odds data and have it look exactly like it does in prod, and then in 
shiny make a selector to either select 1) tonight's games, or 2) the full remaining schedule.


sudo yum install python3.7
sudo yum install httpd-tools
sudo amazon-linux-extras install epel
sudo yum install nginx
sudo pip3 install mlflow

sudo htpasswd -c /etc/nginx/.htpasswd testuser
- password b*gg**

ADD 
    location / {
        proxy_pass http://localhost:5000/;
        auth_basic_user_file /etc/nginx/.htpasswd;
    }

in the server { } block

tracking_uri = "User provides the DNS of the Configured EC2 instance in the step above"
# Sample tracking_uri = http://testuser:test@PUBLIC_DNS_OF_YOUR_EC2

# Set the Tracking URI
mlflow.set_tracking_uri(tracking_uri)
client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

mlflow server --default-artifact-root s3://jacobsbucket97/practice/ --host 0.0.0.0

mlflow server --backend-store-uri sqlite:///mflow.db --default-artifact-root s3://jyablonski-mlflow-bucket/ --host 0.0.0.0

sudo service nginx start
mlflow server --backend-store-uri sqlite:///mflow.db --default-artifact-root s3://jyablonski-mlflow-bucket/ --serve-artifacts --host 0.0.0.0