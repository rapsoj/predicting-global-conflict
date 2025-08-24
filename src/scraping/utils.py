

def generate_search_queries(google_search_templates : list[str], country_names : list[str], search_metrics : list[str], years : list[str]) -> list[dict]:
    '''
    # Outputs
    Array of dictionary with search query and desired country
    '''
    queries_to_search = []
    for search in google_search_templates:
        for country in country_names:
            search_c = search.replace("[country]", country)
            for metric in search_metrics:
                search_m = search_c.replace("[metric]", metric)
                for year in years:
                    search_y = search_m.replace("[year]", year)
                    queries_to_search.append({"search": search_y,
                                         "country": country
                                        })
    return queries_to_search
