"""
Saudi Arabia Oil Well & Field Locations Database
=================================================

Publicly available coordinates for major Saudi Arabian oil fields and well locations.
Compiled from multiple open sources including:
  - Global Energy Monitor (GEM) Oil and Gas Extraction Tracker
  - Wikipedia / Wikidata
  - USGS Open-File Report 97-470-B
  - GetAMap.net geographic database
  - latitude.to / latlong.net coordinate databases
  - Published SPE and AAPG papers
  - Saudi Aramco public disclosures (Aramco Elements Magazine)
  - Saudipedia (official Saudi encyclopedia)

All coordinates are in WGS 84 (EPSG:4326) decimal degrees.
"""

# =============================================================================
# FIELD-LEVEL DATA: Center points, dimensions, and metadata
# =============================================================================

SAUDI_OIL_FIELDS = {

    # =========================================================================
    # 1. GHAWAR FIELD (World's largest conventional oil field)
    # =========================================================================
    "Ghawar": {
        "center": {"lat": 25.213, "lon": 49.2558},
        "wikidata_coords": {"lat": 25.4300, "lon": 49.6200},  # Wikidata Q741935
        "dimensions": {"length_km": 280, "width_km": 30, "area_km2": 8400},
        "type": "Onshore",
        "status": "Operating",
        "discovery_year": 1948,
        "production_start": 1951,
        "operator": "Saudi Aramco",
        "description": "World's largest conventional oil field. N-S trending anticline.",
        "sub_areas": {
            "Fazran": {
                "center": {"lat": 26.10, "lon": 49.35},
                "notes": "Northernmost area. Discovered 1957, production 1962. "
                         "Coordinates estimated from field maps (north of Ain Dar).",
            },
            "Ain_Dar": {
                "center": {"lat": 25.94917, "lon": 49.42378},
                "gosp_coords": {"lat": 25.75, "lon": 49.25},
                "notes": "Discovery well Ain Dar No. 1 drilled 1948, production 1951 at 15,600 bpd.",
            },
            "Shedgum": {
                "center": {"lat": 25.6768, "lon": 49.39743},
                "gas_plant_coords": {"lat": 25.6499, "lon": 49.3906},
                "notes": "Discovered 1952, production 1954. Major gas plant.",
            },
            "Uthmaniyah": {
                "center": {"lat": 25.1938, "lon": 49.3095},
                "gosp_coords": {"lat": 25.1998, "lon": 49.3529},
                "notes": "Discovered 1951, production 1956. Adjacent to Al-Hofuf city.",
            },
            "Hawiyah": {
                "center": {"lat": 24.7983, "lon": 49.18},
                "notes": "Discovery well Hawiyah No. 1 drilled 1953, production 1966. "
                         "Major gas plant at Hawiyah.",
            },
            "Haradh": {
                "center": {"lat": 24.2417, "lon": 49.1864},
                "notes": "Southernmost area. GOSP-1 (1996), GOSP-2 (2003), GOSP-3 (2006). "
                         "300 km SW of Dhahran.",
            },
        },
        "sources": [
            "Global Energy Monitor: https://www.gem.wiki/Ghawar_Oil_and_Gas_Project_(Saudi_Arabia)",
            "Wikipedia: https://en.wikipedia.org/wiki/Ghawar_Field",
            "Wikidata Q741935: https://www.wikidata.org/wiki/Q741935",
            "GetAMap: https://www.getamap.net/maps/saudi_arabia/ash_sharqiyah/_ghawar/",
            "Saudipedia: https://saudipedia.com/en/article/947/economy-and-business/energy-and-natural-resources/ghawar-oil-field",
        ],
    },

    # =========================================================================
    # 2. KHURAIS FIELD (and complex: Abu Jifan, Mazalij)
    # =========================================================================
    "Khurais": {
        "center": {"lat": 25.2028, "lon": 48.0153},
        "wikidata_coords": {"lat": 25.0694, "lon": 48.1950},  # Wikidata Q1639678
        "dimensions": {"length_km": 127, "width_km": 23, "area_km2": 2890},
        "type": "Onshore",
        "status": "Operating",
        "discovery_year": 1957,
        "production_start": 2009,
        "operator": "Saudi Aramco",
        "description": "250 km SW of Dhahran, 150 km ENE of Riyadh. Mega-project.",
        "sub_fields": {
            "Abu_Jifan": {
                "center": {"lat": 25.06311, "lon": 48.03035},
                "area_km2": 520,
                "notes": "SW of main Khurais field.",
            },
            "Mazalij": {
                "center": {"lat": 25.0631, "lon": 48.0304},
                "area_km2": 1630,
                "notes": "SE of Abu Jifan.",
            },
        },
        "sources": [
            "Global Energy Monitor: https://www.gem.wiki/Khurais_Oil_and_Gas_Project_(Saudi_Arabia)",
            "Wikipedia: https://en.wikipedia.org/wiki/Khurais_oil_field",
            "Wikidata Q1639678: https://www.wikidata.org/wiki/Q1639678",
            "GetAMap: https://www.getamap.net/maps/saudi_arabia/ash_sharqiyah/_khurais/",
        ],
    },

    # =========================================================================
    # 3. SHAYBAH FIELD
    # =========================================================================
    "Shaybah": {
        "center": {"lat": 21.7237, "lon": 53.6572},
        "facility_coords": {"lat": 22.5106, "lon": 53.9519},
        "dimensions": {"length_km": 64, "width_km": 13, "area_km2": 832},
        "type": "Onshore",
        "status": "Operating",
        "discovery_year": 1968,
        "production_start": 1998,
        "operator": "Saudi Aramco",
        "description": "Located in Rub' al-Khali (Empty Quarter). Remote desert location.",
        "sources": [
            "Global Energy Monitor: https://www.gem.wiki/Shaybah_Oil_and_Gas_Field_(Saudi_Arabia)",
            "Wikipedia: https://en.wikipedia.org/wiki/Shaybah_oil_field",
            "latlong.net: https://www.latlong.net/place/shaybah-saudi-arabia-796.html",
            "Aramco: https://www.aramco.com/en/what-we-do/mega-projects/shaybah-oil-field-recovery-plant",
        ],
    },

    # =========================================================================
    # 4. ABQAIQ FIELD
    # =========================================================================
    "Abqaiq": {
        "center": {"lat": 26.1687, "lon": 49.7841},
        "wikidata_coords": {"lat": 25.9333, "lon": 49.6667},
        "dimensions": {"length_km": 45, "width_km": 18, "area_km2": 810},
        "type": "Onshore",
        "status": "Operating",
        "discovery_year": 1940,
        "production_start": 1946,
        "operator": "Saudi Aramco",
        "description": "One of oldest Saudi fields. ~60 km SW of Dhahran. Six-reservoir field. "
                       "Major processing facility.",
        "sources": [
            "Global Energy Monitor: https://www.gem.wiki/Abqaiq_Oil_Field_(Saudi_Arabia)",
            "Wikipedia: https://en.wikipedia.org/wiki/Abqaiq_oil_field",
            "GetAMap: https://www.getamap.net/maps/saudi_arabia/saudi_arabia_(general)/_abqaiq/",
            "Saudipedia: https://saudipedia.com/en/article/960/economy-and-business/energy-and-natural-resources/abqaiq-oil-field",
        ],
    },

    # =========================================================================
    # 5. SAFANIYA FIELD (World's largest offshore oil field)
    # =========================================================================
    "Safaniya": {
        "center": {"lat": 28.28, "lon": 48.75},
        "wikipedia_coords": {"lat": 28.2833, "lon": 48.7500},
        "alternate_coords": {"lat": 27.4700, "lon": 48.5200},
        "dimensions": {"length_km": 50, "width_km": 15, "area_km2": 750},
        "type": "Offshore",
        "status": "Operating",
        "discovery_year": 1951,
        "production_start": 1957,
        "operator": "Saudi Aramco",
        "description": "World's largest offshore oil field. ~265 km N of Dhahran. "
                       "160+ unmanned multi-well platforms. 624 wells (as of 1993).",
        "sources": [
            "Global Energy Monitor: https://www.gem.wiki/Safaniya_Oil_and_Gas_Field_(Saudi_Arabia)",
            "Wikipedia: https://en.wikipedia.org/wiki/Safaniya_Oil_Field",
            "latitude.to: https://latitude.to/articles-by-country/sa/saudi-arabia/39179/safaniya-oil-field",
            "Saudipedia: https://saudipedia.com/en/safaniya-oil-field",
        ],
    },

    # =========================================================================
    # 6. BERRI FIELD
    # =========================================================================
    "Berri": {
        "center": {"lat": 27.1145, "lon": 49.6389},
        "getamap_coords": {"lat": 27.2156, "lon": 49.7169},
        "type": "Offshore/Onshore",
        "status": "Operating",
        "discovery_year": 1964,
        "production_start": 1970,
        "operator": "Saudi Aramco",
        "description": "Partly onshore, partly offshore on eastern Saudi coast.",
        "sources": [
            "Global Energy Monitor: https://www.gem.wiki/Berri_Oil_Field_(Saudi_Arabia)",
            "GetAMap: https://www.getamap.net/maps/saudi_arabia/ash_sharqiyah/_berri/",
            "NS Energy: https://www.nsenergybusiness.com/projects/berri-oil-field-expansion/",
        ],
    },

    # =========================================================================
    # 7. ZULUF FIELD
    # =========================================================================
    "Zuluf": {
        "center": {"lat": 28.611, "lon": 49.059},
        "dimensions": {"area_km2": 600},
        "type": "Offshore",
        "status": "Operating",
        "discovery_year": 1965,
        "operator": "Saudi Aramco",
        "description": "Super-giant offshore field. ~240 km N of Dhahran, ~40 km offshore. "
                       "Water depth ~36-40 m. NE-SW anticline.",
        "sources": [
            "Global Energy Monitor: https://www.gem.wiki/Zuluf_Oil_and_Gas_Field_(Saudi_Arabia)",
            "Mapcarta: https://mapcarta.com/12561220",
            "NS Energy: https://www.nsenergybusiness.com/projects/zuluf-oil-field-expansion/",
        ],
    },

    # =========================================================================
    # 8. MANIFA FIELD
    # =========================================================================
    "Manifa": {
        "center": {"lat": 27.7159, "lon": 48.9834},
        "dimensions": {"length_km": 45, "width_km": 18, "area_km2": 810},
        "type": "Offshore (shallow)",
        "status": "Operating",
        "discovery_year": 1957,
        "production_start": 2013,
        "operator": "Saudi Aramco",
        "description": "Off northern Arabian Gulf coast. Water depth 4-6 m. "
                       "One of world's biggest producing fields.",
        "sources": [
            "Global Energy Monitor: https://www.gem.wiki/Manifa_Oil_Field_(Saudi_Arabia)",
            "GetAMap: https://www.getamap.net/maps/saudi_arabia/ash_sharqiyah/_manifa/",
            "Aramco: https://www.aramco.com/en/what-we-do/mega-projects/manifa-offshore-oil-field",
            "Saudipedia: https://saudipedia.com/en/article/1903/economy-and-business/energy-and-natural-resources/manifa-oil-field",
        ],
    },

    # =========================================================================
    # ADDITIONAL FIELDS (bonus)
    # =========================================================================
    "Marjan": {
        "center": {"lat": 28.4389, "lon": 49.6746},
        "type": "Offshore",
        "status": "Operating",
        "operator": "Saudi Aramco",
        "description": "Offshore field in Arabian Gulf.",
        "sources": [
            "Global Energy Monitor: https://www.gem.wiki/Marjan_Oil_Field_(Saudi_Arabia)",
        ],
    },

    "Qatif": {
        "center": {"lat": 26.7077, "lon": 49.9393},
        "type": "Onshore",
        "status": "Operating",
        "operator": "Saudi Aramco",
        "sources": [
            "Global Energy Monitor: https://www.gem.wiki/Qatif_Oil_Field_(Saudi_Arabia)",
        ],
    },

    "Khursaniyah": {
        "center": {"lat": 27.0287, "lon": 48.5437},
        "type": "Onshore",
        "status": "Operating",
        "operator": "Saudi Aramco",
        "sources": [
            "Global Energy Monitor: https://www.gem.wiki/Khursaniyah_Oil_Field_(Saudi_Arabia)",
        ],
    },

    "Dammam": {
        "center": {"lat": 26.4201, "lon": 50.1045},
        "type": "Onshore",
        "status": "Operating",
        "discovery_year": 1938,
        "operator": "Saudi Aramco",
        "description": "First commercial oil discovery in Saudi Arabia.",
        "sources": [
            "Global Energy Monitor: https://www.gem.wiki/Dammam_Oil_Field_(Saudi_Arabia)",
        ],
    },

    "Abu_Safah": {
        "center": {"lat": 26.975, "lon": 50.5522},
        "type": "Offshore",
        "status": "Operating",
        "operator": "Saudi Aramco (joint with Bahrain)",
        "description": "Located in Persian Gulf, SE of Berri field.",
        "sources": [
            "GetAMap: https://www.getamap.net/maps/saudi_arabia/saudi_arabia_(general)/_abusafah/",
            "Wikipedia: https://en.wikipedia.org/wiki/Abu_Safah_field",
        ],
    },

    "Lawhah": {
        "center": {"lat": 28.25272, "lon": 49.61814},
        "type": "Offshore",
        "status": "Operating",
        "operator": "Saudi Aramco",
        "sources": [
            "Global Energy Monitor: https://www.gem.wiki/Lawhah_Oil_Field_(Saudi_Arabia)",
        ],
    },

    "Maharah": {
        "center": {"lat": 28.30833, "lon": 49.48333},
        "type": "Offshore",
        "status": "Operating",
        "operator": "Saudi Aramco",
        "sources": [
            "Global Energy Monitor: https://www.gem.wiki/Maharah_Oil_Field_(Saudi_Arabia)",
        ],
    },

    "Wafra": {
        "center": {"lat": 28.5894, "lon": 47.8886},
        "type": "Onshore",
        "status": "Operating",
        "discovery_year": 1953,
        "operator": "Joint (Saudi-Kuwait Neutral Zone)",
        "description": "Located in Kuwait-Saudi Arabia Neutral Zone.",
        "sources": [
            "Global Energy Monitor: https://www.gem.wiki/Wafra_Oil_Field_(Kuwait-Saudi_Arabia)",
            "Wikipedia: https://en.wikipedia.org/wiki/Wafra_oil_field",
        ],
    },

    "South_Ghawar_Gas": {
        "center": {"lat": 23.18969, "lon": 48.52987},
        "type": "Onshore",
        "status": "Operating",
        "operator": "Saudi Aramco",
        "description": "Gas field in southern Ghawar area.",
        "sources": [
            "Global Energy Monitor: https://www.gem.wiki/South_Ghawar_Gas_Field_(Saudi_Arabia)",
        ],
    },

    "Waqr_Gas": {
        "center": {"lat": 23.96426, "lon": 49.51551},
        "type": "Onshore",
        "status": "Operating",
        "operator": "Saudi Aramco",
        "sources": [
            "Global Energy Monitor: https://www.gem.wiki/Ghawar_Oil_and_Gas_Project_(Saudi_Arabia)",
        ],
    },
}


# =============================================================================
# SPECIFIC WELL LOCATIONS (publicly documented individual wells)
# =============================================================================

SAUDI_OIL_WELLS = [
    # -------------------------------------------------------------------------
    # WELL 1: Dammam No. 7 - "Prosperity Well" (first commercial oil in Saudi Arabia)
    # -------------------------------------------------------------------------
    {
        "well_id": "DAMMAM-7",
        "well_name": "Dammam No. 7 (Prosperity Well)",
        "field": "Dammam",
        "lat": 26.3211,
        "lon": 50.1278,
        "discovery_date": "1938-03-04",
        "status": "Historical (closed 1982)",
        "notes": "First commercial oil discovery in Saudi Arabia. "
                 "Produced 32M+ barrels over 44 years (avg 1,600 bpd).",
        "source": "Wikipedia: https://en.wikipedia.org/wiki/Dammam_No._7",
    },

    # -------------------------------------------------------------------------
    # GHAWAR FIELD - Ain Dar Area Wells
    # -------------------------------------------------------------------------
    {
        "well_id": "AINDAR-1",
        "well_name": "Ain Dar No. 1 (Discovery Well)",
        "field": "Ghawar (Ain Dar)",
        "lat": 25.9492,
        "lon": 49.4238,
        "discovery_date": "1948",
        "status": "Operating (as of 2025, producing 2,800 bpd after 73+ years)",
        "notes": "First Ghawar discovery well. Initial rate 15,600 bpd of dry oil.",
        "source": "GEM: https://www.gem.wiki/Ain_dar_Oil_Field_(Saudi_Arabia) | "
                  "Aramco: https://www.aramco.com/en/news-media/elements-magazine/2025/aramcos-historic-no-1-wells",
    },
    {
        "well_id": "AINDAR-GOSP",
        "well_name": "Ain Dar Gas Oil Separator Plant",
        "field": "Ghawar (Ain Dar)",
        "lat": 25.75,
        "lon": 49.25,
        "status": "Operating Facility",
        "notes": "GOSP facility in Ain Dar area.",
        "source": "GetAMap: https://www.getamap.net/maps/saudi_arabia/ash_sharqiyah/_aindar/",
    },

    # -------------------------------------------------------------------------
    # GHAWAR FIELD - Shedgum Area Wells/Facilities
    # -------------------------------------------------------------------------
    {
        "well_id": "SHEDGUM-CENTER",
        "well_name": "Shedgum Field Center",
        "field": "Ghawar (Shedgum)",
        "lat": 25.6768,
        "lon": 49.3974,
        "status": "Operating",
        "notes": "Center of Shedgum production area. Discovered 1952.",
        "source": "GEM: https://www.gem.wiki/Ghawar_Oil_and_Gas_Project_(Saudi_Arabia)",
    },
    {
        "well_id": "SHEDGUM-GASPLANT",
        "well_name": "Shedgum Gas Plant",
        "field": "Ghawar (Shedgum)",
        "lat": 25.6499,
        "lon": 49.3906,
        "status": "Operating Facility",
        "notes": "Major gas processing plant. Part of Master Gas System since 1975.",
        "source": "Mapcarta: https://mapcarta.com/W37371340 | "
                  "GetAMap: https://www.getamap.net/maps/saudi_arabia/saudi_arabia_(general)/_shedgum/",
    },

    # -------------------------------------------------------------------------
    # GHAWAR FIELD - Uthmaniyah Area Wells/Facilities
    # -------------------------------------------------------------------------
    {
        "well_id": "UTHMANIYAH-GOSP",
        "well_name": "Uthmaniyah GOSP (Gas Oil Separator Plant)",
        "field": "Ghawar (Uthmaniyah)",
        "lat": 25.1998,
        "lon": 49.3529,
        "status": "Operating Facility",
        "notes": "GOSP plant coordinates. Adjacent to Al-Hofuf/Al-Ahsa.",
        "source": "IndustryAbout: https://www.industryabout.com/country-territories-3/1098-saudi-arabia/oil-and-gas/14862-uthmaniyah-11-gosp-plant",
    },

    # -------------------------------------------------------------------------
    # GHAWAR FIELD - Hawiyah Area Wells/Facilities
    # -------------------------------------------------------------------------
    {
        "well_id": "HAWIYAH-1",
        "well_name": "Hawiyah No. 1 (Discovery Well)",
        "field": "Ghawar (Hawiyah)",
        "lat": 24.7983,
        "lon": 49.18,
        "discovery_date": "1953",
        "status": "Historical",
        "notes": "First well in Hawiyah area. Production started 1966.",
        "source": "GetAMap: https://www.getamap.net/maps/saudi_arabia/ash_sharqiyah/_alhawiyah/",
    },

    # -------------------------------------------------------------------------
    # GHAWAR FIELD - Haradh Area Wells/Facilities
    # -------------------------------------------------------------------------
    {
        "well_id": "HARADH-CENTER",
        "well_name": "Haradh Production Area Center",
        "field": "Ghawar (Haradh)",
        "lat": 24.2417,
        "lon": 49.1864,
        "status": "Operating",
        "notes": "GOSP-1 (1996), GOSP-2 (2003), GOSP-3 (2006). "
                 "300 km SW of Dhahran.",
        "source": "GetAMap: https://www.getamap.net/maps/saudi_arabia/ash_sharqiyah/_haradh/ | "
                  "Wikipedia: https://en.wikipedia.org/wiki/Haradh_gas_plant",
    },

    # -------------------------------------------------------------------------
    # KHURAIS FIELD Wells
    # -------------------------------------------------------------------------
    {
        "well_id": "KHURAIS-CENTER",
        "well_name": "Khurais Field Center",
        "field": "Khurais",
        "lat": 25.2028,
        "lon": 48.0153,
        "status": "Operating",
        "notes": "Center of 127 km long field. Mega-project restarted 2009.",
        "source": "GEM: https://www.gem.wiki/Khurais_Oil_and_Gas_Project_(Saudi_Arabia)",
    },
    {
        "well_id": "KHURAIS-GOSP",
        "well_name": "Khurais GOSP Facility",
        "field": "Khurais",
        "lat": 25.0594,
        "lon": 48.0639,
        "status": "Operating Facility",
        "notes": "Gas-oil separator plant at Khurais.",
        "source": "GetAMap: https://www.getamap.net/maps/saudi_arabia/ash_sharqiyah/_khurais/",
    },
    {
        "well_id": "ABUJIFAN-CENTER",
        "well_name": "Abu Jifan Field Center",
        "field": "Khurais Complex (Abu Jifan)",
        "lat": 25.0631,
        "lon": 48.0304,
        "status": "Operating",
        "notes": "520 km2 field SW of Khurais.",
        "source": "GEM: https://www.gem.wiki/Abu_Jifan_Oil_and_Gas_Field_(Saudi_Arabia)",
    },

    # -------------------------------------------------------------------------
    # SHAYBAH FIELD Wells/Facilities
    # -------------------------------------------------------------------------
    {
        "well_id": "SHAYBAH-CENTER",
        "well_name": "Shaybah Field Center",
        "field": "Shaybah",
        "lat": 21.7237,
        "lon": 53.6572,
        "status": "Operating",
        "notes": "GEM center point for Shaybah field in Rub' al-Khali.",
        "source": "GEM: https://www.gem.wiki/Shaybah_Oil_and_Gas_Field_(Saudi_Arabia)",
    },
    {
        "well_id": "SHAYBAH-FACILITY",
        "well_name": "Shaybah Processing Facility / Camp",
        "field": "Shaybah",
        "lat": 22.5106,
        "lon": 53.9519,
        "status": "Operating Facility",
        "notes": "Main processing and residential facility at Shaybah.",
        "source": "latlong.net: https://www.latlong.net/place/shaybah-saudi-arabia-796.html",
    },

    # -------------------------------------------------------------------------
    # ABQAIQ FIELD Wells/Facilities
    # -------------------------------------------------------------------------
    {
        "well_id": "ABQAIQ-CENTER",
        "well_name": "Abqaiq Field Center",
        "field": "Abqaiq",
        "lat": 26.1687,
        "lon": 49.7841,
        "discovery_date": "1940",
        "status": "Operating",
        "notes": "One of the oldest Saudi fields. Major processing hub.",
        "source": "GEM: https://www.gem.wiki/Abqaiq_Oil_Field_(Saudi_Arabia)",
    },
    {
        "well_id": "ABQAIQ-FACILITY",
        "well_name": "Abqaiq Processing Facility",
        "field": "Abqaiq",
        "lat": 25.9371,
        "lon": 49.6776,
        "status": "Operating Facility",
        "notes": "World's largest oil processing facility (stabilization plant).",
        "source": "latitude.to: https://latitude.to/map/sa/saudi-arabia/cities/abqaiq",
    },

    # -------------------------------------------------------------------------
    # SAFANIYA FIELD (Offshore) Wells/Platforms
    # -------------------------------------------------------------------------
    {
        "well_id": "SAFANIYA-CENTER",
        "well_name": "Safaniya Field Center",
        "field": "Safaniya",
        "lat": 28.28,
        "lon": 48.75,
        "status": "Operating",
        "notes": "Center of world's largest offshore field. 624 wells (1993). "
                 "160+ unmanned multi-well platforms.",
        "source": "GEM: https://www.gem.wiki/Safaniya_Oil_and_Gas_Field_(Saudi_Arabia)",
    },
    {
        "well_id": "SAFANIYA-1",
        "well_name": "SFNY-1 (Discovery Well)",
        "field": "Safaniya",
        "lat": 28.2833,
        "lon": 48.7500,
        "discovery_date": "1951",
        "status": "Historical",
        "notes": "First well drilled by Aramco in the Arabian Gulf.",
        "source": "Wikipedia: https://en.wikipedia.org/wiki/Safaniya_Oil_Field | "
                  "Aramco: https://www.aramco.com/en/news-media/elements-magazine/2025/aramcos-historic-no-1-wells",
    },

    # -------------------------------------------------------------------------
    # BERRI FIELD Wells
    # -------------------------------------------------------------------------
    {
        "well_id": "BERRI-CENTER",
        "well_name": "Berri Field Center",
        "field": "Berri",
        "lat": 27.1145,
        "lon": 49.6389,
        "discovery_date": "1964",
        "status": "Operating",
        "notes": "Partly onshore, partly offshore. Production since 1970.",
        "source": "GEM: https://www.gem.wiki/Berri_Oil_Field_(Saudi_Arabia)",
    },
    {
        "well_id": "BERRI-ALT",
        "well_name": "Berri Field (GetAMap reference)",
        "field": "Berri",
        "lat": 27.2156,
        "lon": 49.7169,
        "status": "Operating",
        "notes": "Alternative reference point for Berri field.",
        "source": "GetAMap: https://www.getamap.net/maps/saudi_arabia/ash_sharqiyah/_berri/",
    },

    # -------------------------------------------------------------------------
    # ZULUF FIELD (Offshore) Wells
    # -------------------------------------------------------------------------
    {
        "well_id": "ZULUF-CENTER",
        "well_name": "Zuluf Field Center",
        "field": "Zuluf",
        "lat": 28.611,
        "lon": 49.059,
        "status": "Operating",
        "notes": "Super-giant offshore field. ~600 km2. Water depth ~40 m.",
        "source": "GEM: https://www.gem.wiki/Zuluf_Oil_and_Gas_Field_(Saudi_Arabia)",
    },

    # -------------------------------------------------------------------------
    # MANIFA FIELD (Shallow Offshore) Wells
    # -------------------------------------------------------------------------
    {
        "well_id": "MANIFA-CENTER",
        "well_name": "Manifa Field Center",
        "field": "Manifa",
        "lat": 27.7159,
        "lon": 48.9834,
        "status": "Operating",
        "notes": "Shallow offshore (4-6 m depth). Redeveloped 2013.",
        "source": "GEM: https://www.gem.wiki/Manifa_Oil_Field_(Saudi_Arabia)",
    },

    # -------------------------------------------------------------------------
    # ADDITIONAL FIELD WELLS
    # -------------------------------------------------------------------------
    {
        "well_id": "MARJAN-CENTER",
        "well_name": "Marjan Field Center",
        "field": "Marjan",
        "lat": 28.4389,
        "lon": 49.6746,
        "status": "Operating",
        "notes": "Offshore field in Arabian Gulf.",
        "source": "GEM: https://www.gem.wiki/Marjan_Oil_Field_(Saudi_Arabia)",
    },
    {
        "well_id": "QATIF-CENTER",
        "well_name": "Qatif Field Center",
        "field": "Qatif",
        "lat": 26.7077,
        "lon": 49.9393,
        "status": "Operating",
        "notes": "Onshore field.",
        "source": "GEM: https://www.gem.wiki/Qatif_Oil_Field_(Saudi_Arabia)",
    },
    {
        "well_id": "KHURSANIYAH-CENTER",
        "well_name": "Khursaniyah Field Center",
        "field": "Khursaniyah",
        "lat": 27.0287,
        "lon": 48.5437,
        "status": "Operating",
        "source": "GEM: https://www.gem.wiki/Khursaniyah_Oil_Field_(Saudi_Arabia)",
    },
    {
        "well_id": "DAMMAM-CENTER",
        "well_name": "Dammam Field Center",
        "field": "Dammam",
        "lat": 26.4201,
        "lon": 50.1045,
        "discovery_date": "1938",
        "status": "Operating",
        "notes": "Site of first Saudi oil discovery.",
        "source": "GEM: https://www.gem.wiki/Dammam_Oil_Field_(Saudi_Arabia)",
    },
    {
        "well_id": "ABUSAFAH-CENTER",
        "well_name": "Abu Sa'fah Field Center",
        "field": "Abu Sa'fah",
        "lat": 26.975,
        "lon": 50.5522,
        "status": "Operating",
        "notes": "Offshore. Joint Saudi-Bahrain.",
        "source": "GetAMap: https://www.getamap.net/maps/saudi_arabia/saudi_arabia_(general)/_abusafah/",
    },
    {
        "well_id": "LAWHAH-CENTER",
        "well_name": "Lawhah Field Center",
        "field": "Lawhah",
        "lat": 28.25272,
        "lon": 49.61814,
        "status": "Operating",
        "notes": "Offshore field.",
        "source": "GEM: https://www.gem.wiki/Lawhah_Oil_Field_(Saudi_Arabia)",
    },
    {
        "well_id": "MAHARAH-CENTER",
        "well_name": "Maharah Field Center",
        "field": "Maharah",
        "lat": 28.30833,
        "lon": 49.48333,
        "status": "Operating",
        "notes": "Offshore field.",
        "source": "GEM: https://www.gem.wiki/Maharah_Oil_Field_(Saudi_Arabia)",
    },
    {
        "well_id": "WAFRA-CENTER",
        "well_name": "Wafra Field Center",
        "field": "Wafra",
        "lat": 28.5894,
        "lon": 47.8886,
        "discovery_date": "1953",
        "status": "Operating",
        "notes": "Kuwait-Saudi Neutral Zone.",
        "source": "GEM: https://www.gem.wiki/Wafra_Oil_Field_(Kuwait-Saudi_Arabia)",
    },
    {
        "well_id": "SOUTH-GHAWAR-GAS",
        "well_name": "South Ghawar Gas Field Center",
        "field": "South Ghawar Gas",
        "lat": 23.18969,
        "lon": 48.52987,
        "status": "Operating",
        "notes": "Gas field south of main Ghawar complex.",
        "source": "GEM: https://www.gem.wiki/South_Ghawar_Gas_Field_(Saudi_Arabia)",
    },
    {
        "well_id": "WAQR-GAS",
        "well_name": "Waqr Gas Field Center",
        "field": "Waqr Gas",
        "lat": 23.96426,
        "lon": 49.51551,
        "status": "Operating",
        "source": "GEM: https://www.gem.wiki/Ghawar_Oil_and_Gas_Project_(Saudi_Arabia)",
    },
]


# =============================================================================
# SUMMARY: Flat list of all unique well/location points for quick access
# =============================================================================

ALL_LOCATIONS = [
    # --- ID --- Name ------------------------------------------ Lat ------- Lon ------- Field ----------
    ("DAMMAM-7",            "Dammam No. 7 (Prosperity Well)",   26.3211,    50.1278,    "Dammam"),
    ("AINDAR-1",            "Ain Dar No. 1 (Discovery Well)",   25.9492,    49.4238,    "Ghawar-AinDar"),
    ("AINDAR-GOSP",         "Ain Dar GOSP Facility",            25.75,      49.25,      "Ghawar-AinDar"),
    ("SHEDGUM-CENTER",      "Shedgum Field Center",             25.6768,    49.3974,    "Ghawar-Shedgum"),
    ("SHEDGUM-GASPLANT",    "Shedgum Gas Plant",                25.6499,    49.3906,    "Ghawar-Shedgum"),
    ("UTHMANIYAH-GOSP",     "Uthmaniyah GOSP",                  25.1998,    49.3529,    "Ghawar-Uthmaniyah"),
    ("HAWIYAH-1",           "Hawiyah No. 1",                    24.7983,    49.18,      "Ghawar-Hawiyah"),
    ("HARADH-CENTER",       "Haradh Production Center",         24.2417,    49.1864,    "Ghawar-Haradh"),
    ("KHURAIS-CENTER",      "Khurais Field Center",             25.2028,    48.0153,    "Khurais"),
    ("KHURAIS-GOSP",        "Khurais GOSP Facility",            25.0594,    48.0639,    "Khurais"),
    ("ABUJIFAN-CENTER",     "Abu Jifan Field Center",           25.0631,    48.0304,    "KhuraisComplex"),
    ("SHAYBAH-CENTER",      "Shaybah Field Center",             21.7237,    53.6572,    "Shaybah"),
    ("SHAYBAH-FACILITY",    "Shaybah Processing Facility",      22.5106,    53.9519,    "Shaybah"),
    ("ABQAIQ-CENTER",       "Abqaiq Field Center",              26.1687,    49.7841,    "Abqaiq"),
    ("ABQAIQ-FACILITY",     "Abqaiq Processing Facility",       25.9371,    49.6776,    "Abqaiq"),
    ("SAFANIYA-CENTER",     "Safaniya Field Center",            28.28,      48.75,      "Safaniya"),
    ("SAFANIYA-1",          "SFNY-1 (Discovery Well)",          28.2833,    48.75,      "Safaniya"),
    ("BERRI-CENTER",        "Berri Field Center",               27.1145,    49.6389,    "Berri"),
    ("BERRI-ALT",           "Berri Field (alt ref)",            27.2156,    49.7169,    "Berri"),
    ("ZULUF-CENTER",        "Zuluf Field Center",               28.611,     49.059,     "Zuluf"),
    ("MANIFA-CENTER",       "Manifa Field Center",              27.7159,    48.9834,    "Manifa"),
    ("MARJAN-CENTER",       "Marjan Field Center",              28.4389,    49.6746,    "Marjan"),
    ("QATIF-CENTER",        "Qatif Field Center",               26.7077,    49.9393,    "Qatif"),
    ("KHURSANIYAH-CENTER",  "Khursaniyah Field Center",         27.0287,    48.5437,    "Khursaniyah"),
    ("DAMMAM-CENTER",       "Dammam Field Center",              26.4201,    50.1045,    "Dammam"),
    ("ABUSAFAH-CENTER",     "Abu Sa'fah Field Center",          26.975,     50.5522,    "AbuSafah"),
    ("LAWHAH-CENTER",       "Lawhah Field Center",              28.25272,   49.61814,   "Lawhah"),
    ("MAHARAH-CENTER",      "Maharah Field Center",             28.30833,   49.48333,   "Maharah"),
    ("WAFRA-CENTER",        "Wafra Field Center",               28.5894,    47.8886,    "Wafra"),
    ("SOUTH-GHAWAR-GAS",    "South Ghawar Gas Center",          23.18969,   48.52987,   "SouthGhawar"),
    ("WAQR-GAS",            "Waqr Gas Field Center",            23.96426,   49.51551,   "WaqrGas"),
    ("FAZRAN-EST",          "Fazran Area (estimated)",           26.10,      49.35,      "Ghawar-Fazran"),
    ("GHAWAR-PROJECT",      "Ghawar Project Center (GEM)",      25.213,     49.2558,    "Ghawar"),
    ("MAZALIJ-CENTER",      "Mazalij Field Center",             25.0631,    48.0304,    "KhuraisComplex"),
]

# Total: 34 unique location points


# =============================================================================
# HELPER: Export as CSV-ready format
# =============================================================================

def to_csv_rows():
    """Return all locations as a list of dicts suitable for CSV export."""
    rows = []
    for loc_id, name, lat, lon, field in ALL_LOCATIONS:
        rows.append({
            "id": loc_id,
            "name": name,
            "latitude": lat,
            "longitude": lon,
            "field": field,
        })
    return rows


def print_summary():
    """Print a formatted summary of all locations."""
    print(f"{'ID':<25} {'Name':<40} {'Lat':>10} {'Lon':>10} {'Field':<20}")
    print("-" * 110)
    for loc_id, name, lat, lon, field in ALL_LOCATIONS:
        print(f"{loc_id:<25} {name:<40} {lat:>10.5f} {lon:>10.5f} {field:<20}")
    print(f"\nTotal locations: {len(ALL_LOCATIONS)}")


if __name__ == "__main__":
    print_summary()
