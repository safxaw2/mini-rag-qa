from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api.main import app
from src.api import config
from src.api.routers import document as document_router


def test_document_routes_use_api_prefix_once():
    client = TestClient(app)

    response = client.get("/api/supported-extensions")

    assert response.status_code == 200
    assert set(response.json()["formats"]) == {
        ".pdf",
        ".html",
        ".htm",
        ".md",
        ".markdown",
        ".docx",
    }

    paths = {route.path for route in app.routes}
    assert "/api/upload" in paths
    assert "/api/parse-path" in paths
    assert "/api/api/upload" not in paths


def test_list_collections_returns_existing_collection_names(monkeypatch):
    monkeypatch.setattr(
        document_router.ChromaVectorStore,
        "list_collections",
        lambda persist_directory: ["documents", "reports"],
    )

    response = TestClient(app).get("/api/collections")

    assert response.status_code == 200
    assert response.json() == {"collections": ["documents", "reports"]}


def test_create_collection(monkeypatch):
    created = {}

    class StubStore:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(document_router, "ChromaVectorStore", StubStore)

    response = TestClient(app).post(
        "/api/collections",
        json={"collection_name": " new_collection "},
    )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "collection_name": "new_collection",
    }
    assert created["collection_name"] == "new_collection"
    assert created["search_ef"] == config.settings.VECTOR_HNSW_SEARCH_EF


def test_create_collection_rejects_empty_name():
    response = TestClient(app).post(
        "/api/collections",
        json={"collection_name": "   "},
    )

    assert response.status_code == 400


def test_upload_stores_parsed_chunks_in_requested_collection(monkeypatch, tmp_path):
    stored = {}

    class StubStore:
        def __init__(self, embedding_model, collection_name):
            stored["collection_name"] = collection_name

        def count_file_hash(self, file_hash, namespace):
            stored["file_hash"] = file_hash
            return 0

        def add_parse_result(self, parse_result, name_space):
            stored["file_name"] = parse_result.file_name
            stored["namespace"] = name_space
            return ["chunk-1"]

    parser = SimpleNamespace(
        parse_from_bytes=lambda contents, filename: SimpleNamespace(
            file_name=filename,
            total_chunks=1,
        )
    )
    monkeypatch.setattr(config.settings, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(document_router.ParserFactory, "get_parser", lambda filename: parser)
    monkeypatch.setattr(
        document_router,
        "get_document_store",
        lambda collection_name: StubStore(object(), collection_name),
    )

    response = TestClient(app).post(
        "/api/upload",
        params={"collection_name": "test_collection"},
        files={"file": ("sample.md", b"sample content", "text/markdown")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "file_name": "sample.md",
        "stored": True,
        "doc_count": 1,
    }
    assert stored == {
        "collection_name": "test_collection",
        "file_hash": "571ca3b4ef92a81f8c062f2c2437b9116435d1575589a7b64a5c607d058fde0d",
        "file_name": "sample.md",
        "namespace": "test_collection",
    }


def test_upload_skips_parsing_and_storage_for_duplicate_content(monkeypatch, tmp_path):
    class ExistingStore:
        def count_file_hash(self, file_hash, namespace):
            assert len(file_hash) == 64
            assert namespace == "documents"
            return 3

    monkeypatch.setattr(config.settings, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(
        document_router,
        "get_document_store",
        lambda collection_name: ExistingStore(),
    )
    monkeypatch.setattr(
        document_router.ParserFactory,
        "get_parser",
        lambda filename: (_ for _ in ()).throw(AssertionError("duplicate must not parse")),
    )

    response = TestClient(app).post(
        "/api/upload",
        params={"collection_name": "documents"},
        files={"file": ("sample.md", b"already stored", "text/markdown")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "file_name": "sample.md",
        "stored": True,
        "duplicate": True,
        "doc_count": 3,
    }
    assert list(tmp_path.iterdir()) == []
