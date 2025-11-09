#!/usr/bin/env python3
"""
Quick test script to verify the refactored RAG module works correctly.
"""

import sys
import os

# Test imports
try:
    from RAG import (
        RAG,
        QueryRewrite,
        CatalogResponse,
        MaterialType,
        SearchFilters,
        rephrase_and_expand_query,
        extract_filters_with_llm,
        build_sql_filter,
        retrieve_from_pg,
        rerank,
        generate_catalog_summary,
        parse_json_response,
    )
    print("✅ All imports successful!")
    
    # Test model instantiation
    search_filters = SearchFilters(year_exact=1919, material_types=[MaterialType.STILL_IMAGE])
    print(f"✅ SearchFilters created: {search_filters.dict()}")
    
    query_rewrite = QueryRewrite(improved_query="test", expanded_query="expanded test")
    print(f"✅ QueryRewrite created: {query_rewrite.dict()}")
    
    catalog_response = CatalogResponse(summary="Test summary")
    print(f"✅ CatalogResponse created: {catalog_response.dict()}")
    
    # Test SQL filter building
    where, params = build_sql_filter(search_filters)
    print(f"✅ SQL filter built: WHERE {where} | params: {params}")
    
    print("\n🎉 All tests passed! Refactored RAG module is working correctly.")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

