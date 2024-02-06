import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

def send_email(subject, body, to_addr, from_addr, password, filename, filepath):
    # Set up the MIME
    msg = MIMEMultipart()
    msg['From'] = from_addr
    msg['To'] = to_addr
    msg['Subject'] = subject

    # Attach the body with the msg instance
    msg.attach(MIMEText(body, 'plain'))

    # Open the file to be sent
    attachment = open(filepath, "rb")

    # Instance of MIMEBase and named as part
    part = MIMEBase('application', 'octet-stream')

    # To change the payload into encoded form
    part.set_payload((attachment).read())

    # Encode into base64
    encoders.encode_base64(part)

    part.add_header('Content-Disposition', "attachment; filename= %s" % filename)

    # Attach the instance 'part' to instance 'msg'
    msg.attach(part)

    # Create SMTP session for sending the mail
    session = smtplib.SMTP('smtp.163.com', 25)  # use 163 mail with port 25
    session.starttls()  # enable security
    session.login(from_addr, password)  # login with mail_id and password

    text = msg.as_string()
    session.sendmail(from_addr, to_addr, text)
    session.quit()

    print('Mail Sent')

if __name__ == '__main__':
    send_email(
        subject=f"subject",
        body="body",
        to_addr='jechin@qq.com',
        from_addr='jechinyu@163.com',
        password='RLAUAWTBXQSLMZQL',
        filename='a.png',
        filepath='/root/miccai/RSNA-ICH/research/dept8/qdou/data/RSNA-ICH/organized/a.png'
    )