"""
Клиент для взаимодействия с Python RAG Server
Простая реализация HTTP клиента
"""

import aiohttp
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class RAGClient:
    """HTTP клиент для RAG сервера"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Получить HTTP сессию"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def upload_file(self, file_path: str, rag_server_url: str) -> Dict[str, Any]:
        """
        Загружает и немедленно обрабатывает файл в RAG сервере, используя
        атомарный эндпоинт. Отправляет содержимое файла через HTTP POST.
        
        Args:
            file_path: Путь к файлу на локальном хосте.
            rag_server_url: URL RAG сервера.
            
        Returns:
            Результат обработки от сервера в формате JSON.
        """
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        if not file_path_obj.is_file():
            raise ValueError(f"Указанный путь не является файлом: {file_path}")
        
        # Проверка расширения файла на стороне клиента
        allowed_extensions = {
            ".txt", ".md", ".pdf", ".py", ".js", ".java", 
            ".c", ".cpp", ".json", ".yaml", ".yml", ".ini", ".toml"
        }
        
        if file_path_obj.suffix.lower() not in allowed_extensions:
            raise ValueError(f"Неподдерживаемый тип файла: {file_path_obj.suffix}")
        
        session = await self._get_session()
        
        try:
            with open(file_path_obj, 'rb') as f:
                data = aiohttp.FormData()
                # Сервер ожидает поле 'file'
                data.add_field('file', f, filename=file_path_obj.name, content_type='application/octet-stream')
                
                # Используем новый атомарный эндпоинт
                upload_url = f"{rag_server_url.rstrip('/')}/api/upload"
                
                async with session.post(upload_url, data=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ошибка загрузки и обработки файла: HTTP {response.status}, {error_text}")
                    
                    result = await response.json()
                    if result.get("status") != "success":
                        raise Exception(f"Ошибка на сервере: {result.get('error', 'Unknown error')}")

                    logger.info(f"Файл {file_path_obj.name} успешно загружен и обработан.")
                    
                    # Добавляем информацию о файле в результат для полноты
                    result['source_file_name'] = file_path_obj.name
                    result['source_file_size'] = file_path_obj.stat().st_size
                    return result
            
        except Exception as e:
            logger.error(f"Ошибка при работе с файлом {file_path}: {e}")
            raise # Передаем исключение выше для обработки в MCP сервере
    
    async def search(
        self, 
        query: str, 
        top_k: int = 5,
        search_type: str = "hybrid",
        use_reranker: bool = True,
        expand_query: bool = False,
        filters: Optional[Dict[str, Any]] = None,
        rag_server_url: str = "http://host.docker.internal:8000",
    ) -> List[Dict[str, Any]]:
        """
        Выполнить поиск через JSON API
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            search_type: Тип поиска ('hybrid', 'semantic', 'keyword')
            use_reranker: Использовать ли переранжирование
            expand_query: Использовать ли расширение запроса
            rag_server_url: URL RAG сервера
            
        Returns:
            Список результатов поиска
        """
        session = await self._get_session()
        try:
            # Используем новый JSON API endpoint
            search_url = f"{rag_server_url.rstrip('/')}/api/search"
            
            # Отправляем запрос с параметрами (boolean конвертируем в строки)
            params = {
                "query": query,
                "top_k": top_k,
                "search_type": search_type,
                "use_reranker": str(use_reranker).lower(),
                "expand_query": str(expand_query).lower(),
            }
            for _k in ("project", "report_type", "type", "tags", "date_from", "date_to"):
                if filters and filters.get(_k):
                    _v = filters[_k]
                    params[_k] = ",".join(_v) if isinstance(_v, list) else _v
            
            async with session.post(search_url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Ошибка поиска: HTTP {response.status}, {await response.text()}")
                
                result = await response.json()
                
                # Проверяем наличие ошибки в ответе
                if "error" in result:
                    raise Exception(f"Ошибка сервера: {result['error']}")
                
                logger.info(f"Поиск выполнен для запроса: '{query}', найдено: {result.get('total_results', 0)} результатов")
                
                return result.get("results", [])
                
        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            raise
    
    async def ingest_report(
        self,
        path: str,
        project: Optional[str] = None,
        report_type: Optional[str] = None,
        report_date: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Any] = None,
        doc_type: Optional[str] = None,
        rag_server_url: str = "http://host.docker.internal:8000",
    ) -> Dict[str, Any]:
        """Ingest a report file (server-accessible path) as an OKF doc via /api/ingest-report."""
        session = await self._get_session()
        url = f"{rag_server_url.rstrip('/')}/api/ingest-report"
        params = {"path": path}
        if isinstance(tags, list):
            tags = ",".join(tags)
        for _k, _v in (("project", project), ("report_type", report_type),
                       ("report_date", report_date), ("title", title),
                       ("description", description), ("tags", tags), ("type", doc_type)):
            if _v:
                params[_k] = _v
        async with session.post(url, params=params) as response:
            result = await response.json()
            if response.status != 200:
                raise Exception(f"ingest-report failed: HTTP {response.status}, {result}")
            logger.info(f"Ingested report {path} (project={project})")
            return result

    async def delete_report(
        self,
        source_path: str,
        rag_server_url: str = "http://host.docker.internal:8000",
    ) -> Dict[str, Any]:
        """Delete all chunks for a report by its source_path via /api/delete-report."""
        session = await self._get_session()
        url = f"{rag_server_url.rstrip('/')}/api/delete-report"
        async with session.delete(url, params={"source_path": source_path}) as response:
            result = await response.json()
            if response.status != 200:
                raise Exception(f"delete-report failed: HTTP {response.status}, {result}")
            logger.info(f"Deleted report {source_path} ({result.get('deleted_chunks')} chunks)")
            return result

    async def close(self):
        """Закрыть HTTP сессию"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close() 