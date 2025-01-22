# Digital Commonwealth API Guide

Basic information on available APIs and creating requests can be found here: https://www.digitalcommonwealth.org/api.

The sections below provide more detail on fetching metadata records and associated IIIF manifests, binary files, OCR text, etc.

## Content architecture
APIs can be used to retrive information about:
1. **Items**, which are things like books, photographs, maps, newspaper issues, paintings, audio recordings, films, etc.
2. **Files**, which are binary objects like images, text files, audio files, video files, etc.

## Fetching metadata records
Use the JSON API to request a list of records, or the full metadata record for an item. A record contains the relevant information about an item, like its title, creator, date, topical coverage, format, etc.

There is a list of fields and their defintions here: [SolrDocument field reference: public API](https://github.com/boston-library/solr-core-conf/wiki/SolrDocument-field-reference:-public-API)

#### List of records
Any search that can be performed in the [front-end UI](https://www.digitalcommonwealth.org) can be turned into an API query by replacing `/search` with `/search.json` in the request URL.

List of records matching keyword "Boston":
```
https://www.digitalcommonwealth.org/search.json?q=Boston&search_field=all_fields
```
List of records matching keyword "Boston" and format="Maps":
```
https://www.digitalcommonwealth.org/search.json?f%5Bgenre_basic_ssim%5D%5B%5D=Maps&q=Boston&search_field=all_fields
```
List all records:
```
https://www.digitalcommonwealth.org/search.json?search_field=all_fields&q=
```

Each API call returns 20 records by default, this can be adjusted by setting `per_page` in the query params (100 max). Iterate over the list of pages to get all the records.

#### Full metadata for an item

To retrieve the full metadata record for an individual item as JSON, append `.json` to the page url, as in the examples below:
```
# normal, return HTML
https://www.digitalcommonwealth.org/search/commonwealth:abcd12345

# return JSON
https://www.digitalcommonwealth.org/search/commonwealth:abcd12345.json
```

The _canonical URL_ for an item is found in the `identifier_uri_ss` value from the JSON metadata record, it looks like:
```
https://ark.digitalcommonwealth.org/ark:/50959/:id
```

## Fetching image content using the IIIF Manifest

To retrieve image content asociated with an item, use the item's [IIIF Presentation Manifest](https://iiif.io/api/presentation/2.1/) to get a list of associated images.

The IIIF manifest URL is found in the `identifier_iiif_manifest_ss` field in the JSON metadata record.

Alternately, using the canonical URL (found in the `identifier_uri_ss` value from the JSON metadata record), it is possible to construct a URL that will return a IIIF Presentation manifest for the item by appending `/manifest` to the URL. This will provide a list of the component images for the item, in order, for example:
```
https://ark.digitalcommonwealth.org/ark:/50959/abcd12345/manifest
```
(If no manifest exists, the URL will return a 404.)

In the manifest, there is a `"sequences"` key that provides a list of the item's image files ("canvases"). For each canvas, use the `["images"]["resource"]["service"]["@id"]` path to obtain the base IIIF image URL. This will look something like:
```
https://iiif.digitalcommonwealth.org/iiif/2/commonwealth:zyxw98765
```
You can use the IIIF Image API to obtain any size image you want (see the IIIF [Image API 2.1.1](https://iiif.io/api/image/2.1/) guide for more information). For example, to obtain the full size image in JPEG format, use this syntax:
```
https://iiif.digitalcommonwealth.org/iiif/2/commonwealth:zyxw98765/full/full/0/default.jpg
```

## Fetching text content for an item

A plain-text serialization of the transcribed text for an item (if it exists) can be returned by appending `/text` to the canonical URL (found in the `identifier_uri_ss` value from the JSON metadata record), for example:
```
https://ark.digitalcommonwealth.org/ark:/50959/abcd12345/text
```
(If no text is available the URL will return a 404.)

