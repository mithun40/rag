import os
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

def upload_and_analyze_document(
    file_path: str,
    account_name: str,
    account_key: str,
    container_name: str,
    doc_intel_endpoint: str,
    doc_intel_key: str
):
    """
    Uploads a document to Azure Blob Storage and analyzes it using Document Intelligence.

    Args:
        file_path (str): Path to the .pdf or .docx file.
        account_name (str): Azure Storage account name.
        account_key (str): Azure Storage account key.
        container_name (str): Name of the blob container.
        doc_intel_endpoint (str): Document Intelligence endpoint URL.
        doc_intel_key (str): Document Intelligence API key.

    Returns:
        Analysis result object.
    """
    # Create connection string
    connection_string = (
        f"DefaultEndpointsProtocol=https;"
        f"AccountName={account_name};"
        f"AccountKey={account_key};"
        f"EndpointSuffix=core.windows.net"
    )

    # Extract blob name from file path
    blob_name = os.path.basename(file_path)

    # Upload file to Blob Storage
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    with open(file_path, "rb") as data:
        container_client.upload_blob(name=blob_name, data=data, overwrite=True)

    # Generate SAS URL for the uploaded blob (valid for 1 hour)
    sas_token = generate_blob_sas(
        account_name=account_name,
        container_name=container_name,
        blob_name=blob_name,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=1)
    )
    blob_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"

    # Analyze document using Document Intelligence
    client = DocumentIntelligenceClient(doc_intel_endpoint, AzureKeyCredential(doc_intel_key))
    poller = client.begin_analyze_document_from_url("prebuilt-document", blob_url)
    result = poller.result()
    return result

# Example usage:
# result = upload_and_analyze_document(
#     file_path="sample.pdf",
#     account_name="your_account_name",
#     account_key="your_account_key",
#     container_name="your_container",
#     doc_intel_endpoint="https://your-doc-intel-endpoint.cognitiveservices.azure.com/",
#     doc_intel_key="your_doc_intel_key"
# )
# print(result)
