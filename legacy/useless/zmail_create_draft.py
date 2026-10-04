from __future__ import print_function

import base64
from email.message import EmailMessage

import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import os
from google.oauth2 import service_account

# If modifying these scopes, delete the file token.json.
# https://developers.google.com/gmail/api/auth/scopes para vver todas las SCOPES
SCOPES = ['https://mail.google.com/' ] #https://www.googleapis.com/auth/gmail.readonly','https://www.googleapis.com/auth/gmail.send']
#SCOPES = ['https://www.googleapis.com/auth/gmail.readonly','https://www.googleapis.com/auth/gmail.send']
token_credential = "keys/token.json" # "keys/SendMail_test_CRE.json"


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

def gmail_create_draft():
    """Create and insert a draft email.
       Print the returned draft's message and id.
       Returns: Draft object, including draft id and message meta data.

      Load pre-authorized user credentials from the environment.
      TODO(developer) - See https://developers.google.com/identity
      for guides on implementing OAuth2 for the application.
    """
    #creds, _ = google.auth.default()

    creds = get_Oauth_google_crede()

    try:
        # create gmail api client
        service = build('gmail', 'v1', credentials=creds)

        message = EmailMessage()

        message.set_content('This is automated draft mail')

        message['To'] = 'gduser1@workspacesamples.dev'
        message['From'] = 'gduser2@workspacesamples.dev'
        message['Subject'] = 'Automated draft'

        # encoded message
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        create_message = {
            'message': {
                'raw': encoded_message
            }
        }
        # pylint: disable=E1101
        draft = service.users().drafts().create(userId="me",
                                                body=create_message).execute()

        print(F'Draft id: {draft["id"]}\nDraft message: {draft["message"]}')

    except HttpError as error:
        print(F'An error occurred: {error}')
        draft = None

    return draft


if __name__ == '__main__':
    gmail_create_draft()