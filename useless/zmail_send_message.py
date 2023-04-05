from __future__ import print_function

import base64
from email.message import EmailMessage
from email.mime.text import MIMEText



import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
# https://developers.google.com/gmail/api/auth/scopes para vver todas las SCOPES
# En caso de caducar expirar crear cuanta de servicio https://console.developers.google.com/iam-admin/serviceaccounts NO es eso
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly' ] #https://www.googleapis.com/auth/gmail.readonly','https://www.googleapis.com/auth/gmail.send']
#SCOPES = ['https://www.googleapis.com/auth/gmail.readonly','https://www.googleapis.com/auth/gmail.send']
token_credential = "keys/token.json" # "keys/SendMail_test_CRE.json"

import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
def get_Oauth_google_crede():
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(token_credential):
        creds = Credentials.from_authorized_user_file(token_credential, SCOPES)
        print("In case of: missing fields refresh_token, client_secret, client_id. borrar fichero token.json , para que se refresque automaticamente Ruta: "+token_credential )
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "keys/SendMail_test_CRE.json", SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(token_credential, 'w') as token:
            token.write(creds.to_json())
    return creds



# https://developers.google.com/gmail/api/guides/sending
def gmail_send_message():
    """Create and send an email message
    Print the returned  message id
    Returns: Message object, including message id

    Load pre-authorized user credentials from the environment.
    TODO(developer) - See https://developers.google.com/identity
    for guides on implementing OAuth2 for the application.
    """
    #creds, _ = google.auth.default()
    creds = get_Oauth_google_crede()

    try:
        service = build('gmail', 'v1', credentials=creds)
        # message = EmailMessage()
        # message.set_content('<br>This is <b>automated<b> draft mail<br><br><br>END 3')
        f_html_mail = open("zmail_template.html", "r")
        text_html_mail = f_html_mail.read()

        message = MIMEText(text_html_mail,'html') #MIMEText('<br>This is <b>automated<b> draft mail<br><br><br>END 3','html')

        message['To'] = 'example@gmail.com' # 'prueba19j97@gmail.com' #gduser1@workspacesamples.dev'
        message['From'] = 'example@workspacesamples.dev'
        message['Subject'] = 'Alert System Whale Hunter Trading'

        # encoded message
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode() #  base64.urlsafe_b64encode(message.as_bytes()).decode()
        #encoded_message = base64.urlsafe_b64encode(message.as_string())
        #{'raw': base64.urlsafe_b64encode(message.as_string()), 'payload': {'mimeType': 'text/html'}}

        create_message = {
            'raw': encoded_message
            #'payload': {'mimeType': 'text/html'} #TO send HTML
        }
        # pylint: disable=E1101
        send_message = (service.users().messages().send
                        (userId="me", body=create_message).execute())
        print(F'Message Id: {send_message["id"]}')
    except HttpError as error:
        print(F'An error occurred: {error}')
        send_message = None
    return send_message


if __name__ == '__main__':
    gmail_send_message()