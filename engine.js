
//  function cards(a, b, c, d, e, f) {
//   var type = {
//   card: 
  
//     '<div class='+a+'>'+
//   '<div class="card bg-light mb-3">'+
//     '<div class="card-body">'+
//       '<h5 class="card-title">Campaign Finance Reform</h5>'+
//       '<p1 class="card-text">The ability of special interests to corrupt our democracy must end.</p1>'+
//       `<a id="CampaignFinanceReform" class="align-self-end btn btn-sm btn-block btn-primary" onclick= "zz('Campaign Finance Reform')">Engage Now</a>`+
//     '</div>'+
//   '</div>'+
// '</div>'
//  };
{/* <div class="col-sm-6">
              <div class="card bg-light mb-3">
                <div class="card-body">
                  <h5 class="card-title">Campaign Finance Reform</h5>
                  <p1 class="card-text">The ability of special interests to corrupt our democracy must end.</p1>
                  <a id="CampaignFinanceReform" class="align-self-end btn btn-sm btn-block btn-primary" onclick= "zz('Campaign Finance Reform')">Engage Now</a>
                </div>
              </div>
            </div> */}
// }






var industrycodes = 
[ 
  {
    "Catcode": "A11",
    "Catname": "A0000",
    "Catorder": "Agriculture",
    "Industry": "Misc Agriculture",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "A01",
    "Catname": "A1000",
    "Catorder": "Crop production & basic processing",
    "Industry": "Crop Production & Basic Processing",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "A01",
    "Catname": "A1100",
    "Catorder": "Cotton",
    "Industry": "Crop Production & Basic Processing",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "A01",
    "Catname": "A1200",
    "Catorder": "Sugar cane & sugar beets",
    "Industry": "Crop Production & Basic Processing",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "A02",
    "Catname": "A1300",
    "Catorder": "Tobacco & Tobacco products",
    "Industry": "Tobacco",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "A01",
    "Catname": "A1400",
    "Catorder": "Vegetables, fruits and tree nut",
    "Industry": "Crop Production & Basic Processing",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "A01",
    "Catname": "A1500",
    "Catorder": "Wheat, corn, soybeans and cash grain",
    "Industry": "Crop Production & Basic Processing",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "A01",
    "Catname": "A1600",
    "Catorder": "Other commodities (incl rice, peanuts, honey)",
    "Industry": "Crop Production & Basic Processing",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "A04",
    "Catname": "A2000",
    "Catorder": "Milk & dairy producers",
    "Industry": "Dairy",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "A05",
    "Catname": "A2300",
    "Catorder": "Poultry & eggs",
    "Industry": "Poultry & Eggs",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "A06",
    "Catname": "A3000",
    "Catorder": "Livestock",
    "Industry": "Livestock",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "A07",
    "Catname": "A3100",
    "Catorder": "Animal feed & health products",
    "Industry": "Agricultural Services/Products",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "A06",
    "Catname": "A3200",
    "Catorder": "Sheep and Wool Producers",
    "Industry": "Livestock",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "A06",
    "Catname": "A3300",
    "Catorder": "Feedlots & related livestock services",
    "Industry": "Livestock",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "A06",
    "Catname": "A3500",
    "Catorder": "Horse breeders",
    "Industry": "Livestock",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "A07",
    "Catname": "A4000",
    "Catorder": "Agricultural services & related industries",
    "Industry": "Agricultural Services/Products",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "A07",
    "Catname": "A4100",
    "Catorder": "Agricultural chemicals (fertilizers & pesticides)",
    "Industry": "Agricultural Services/Products",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "A07",
    "Catname": "A4200",
    "Catorder": "Farm machinery & equipment",
    "Industry": "Agricultural Services/Products",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "A07",
    "Catname": "A4300",
    "Catorder": "Grain traders & terminals",
    "Industry": "Agricultural Services/Products",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "A07",
    "Catname": "A4500",
    "Catorder": "Veterinarians",
    "Industry": "Agricultural Services/Products",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "A10",
    "Catname": "A5000",
    "Catorder": "Forestry & Forest Products",
    "Industry": "Forestry & Forest Products",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "A10",
    "Catname": "A5200",
    "Catorder": "Paper & pulp mills and paper manufacturing",
    "Industry": "Forestry & Forest Products",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "A07",
    "Catname": "A6000",
    "Catorder": "Farm organizations & cooperatives",
    "Industry": "Agricultural Services/Products",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "A07",
    "Catname": "A6500",
    "Catorder": "Farm bureaus",
    "Industry": "Agricultural Services/Products",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "A07",
    "Catname": "A8000",
    "Catorder": "Florists & Nursery Services",
    "Industry": "Agricultural Services/Products",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "C01",
    "Catname": "B0000",
    "Catorder": "Construction & Public Works",
    "Industry": "General Contractors",
    "Sector": "Construction",
    "Sector Long": "Construction"
  },
  {
    "Catcode": "C01",
    "Catname": "B0500",
    "Catorder": "Builders associations",
    "Industry": "General Contractors",
    "Sector": "Construction",
    "Sector Long": "Construction"
  },
  {
    "Catcode": "C01",
    "Catname": "B1000",
    "Catorder": "Public works, industrial & commercial construction",
    "Industry": "General Contractors",
    "Sector": "Construction",
    "Sector Long": "Construction"
  },
  {
    "Catcode": "C01",
    "Catname": "B1200",
    "Catorder": "Dredging Contractors",
    "Industry": "General Contractors",
    "Sector": "Construction",
    "Sector Long": "Construction"
  },
  {
    "Catcode": "C01",
    "Catname": "B1500",
    "Catorder": "Construction, unclassified",
    "Industry": "General Contractors",
    "Sector": "Construction",
    "Sector Long": "Construction"
  },
  {
    "Catcode": "C02",
    "Catname": "B2000",
    "Catorder": "Residential construction",
    "Industry": "Home Builders",
    "Sector": "Construction",
    "Sector Long": "Construction"
  },
  {
    "Catcode": "C02",
    "Catname": "B2400",
    "Catorder": "Mobile home construction",
    "Industry": "Home Builders",
    "Sector": "Construction",
    "Sector Long": "Construction"
  },
  {
    "Catcode": "C03",
    "Catname": "B3000",
    "Catorder": "Special trade contractors",
    "Industry": "Special Trade Contractors",
    "Sector": "Construction",
    "Sector Long": "Construction"
  },
  {
    "Catcode": "C03",
    "Catname": "B3200",
    "Catorder": "Electrical contractors",
    "Industry": "Special Trade Contractors",
    "Sector": "Construction",
    "Sector Long": "Construction"
  },
  {
    "Catcode": "C03",
    "Catname": "B3400",
    "Catorder": "Plumbing, heating & air conditioning",
    "Industry": "Special Trade Contractors",
    "Sector": "Construction",
    "Sector Long": "Construction"
  },
  {
    "Catcode": "C03",
    "Catname": "B3600",
    "Catorder": "Landscaping & Excavation Svcs",
    "Industry": "Special Trade Contractors",
    "Sector": "Construction",
    "Sector Long": "Construction"
  },
  {
    "Catcode": "C04",
    "Catname": "B4000",
    "Catorder": "Engineering, architecture & construction mgmt svcs",
    "Industry": "Construction Services",
    "Sector": "Construction",
    "Sector Long": "Construction"
  },
  {
    "Catcode": "C04",
    "Catname": "B4200",
    "Catorder": "Architectural services",
    "Industry": "Construction Services",
    "Sector": "Construction",
    "Sector Long": "Construction"
  },
  {
    "Catcode": "C04",
    "Catname": "B4300",
    "Catorder": "Surveying",
    "Industry": "Construction Services",
    "Sector": "Construction",
    "Sector Long": "Construction"
  },
  {
    "Catcode": "C04",
    "Catname": "B4400",
    "Catorder": "Engineers - type unknown",
    "Industry": "Construction Services",
    "Sector": "Construction",
    "Sector Long": "Construction"
  },
  {
    "Catcode": "C05",
    "Catname": "B5000",
    "Catorder": "Building materials",
    "Industry": "Building Materials & Equipment",
    "Sector": "Construction",
    "Sector Long": "Construction"
  },
  {
    "Catcode": "C05",
    "Catname": "B5100",
    "Catorder": "Stone, clay, glass & concrete products",
    "Industry": "Building Materials & Equipment",
    "Sector": "Construction",
    "Sector Long": "Construction"
  },
  {
    "Catcode": "C05",
    "Catname": "B5200",
    "Catorder": "Lumber and wood products",
    "Industry": "Building Materials & Equipment",
    "Sector": "Construction",
    "Sector Long": "Construction"
  },
  {
    "Catcode": "C05",
    "Catname": "B5300",
    "Catorder": "Plumbing & pipe products",
    "Industry": "Building Materials & Equipment",
    "Sector": "Construction",
    "Sector Long": "Construction"
  },
  {
    "Catcode": "C05",
    "Catname": "B5400",
    "Catorder": "Other construction-related products",
    "Industry": "Building Materials & Equipment",
    "Sector": "Construction",
    "Sector Long": "Construction"
  },
  {
    "Catcode": "C05",
    "Catname": "B5500",
    "Catorder": "Electrical Supply",
    "Industry": "Building Materials & Equipment",
    "Sector": "Construction",
    "Sector Long": "Construction"
  },
  {
    "Catcode": "C05",
    "Catname": "B6000",
    "Catorder": "Construction equipment",
    "Industry": "Building Materials & Equipment",
    "Sector": "Construction",
    "Sector Long": "Construction"
  },
  {
    "Catcode": "B00",
    "Catname": "C0000",
    "Catorder": "Communications & Electronics",
    "Industry": "Misc Communications/Electronics",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B01",
    "Catname": "C1000",
    "Catorder": "Printing and publishing (printed & online)",
    "Industry": "Printing & Publishing",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B01",
    "Catname": "C1100",
    "Catorder": "Book, newspaper & periodical publishing",
    "Industry": "Printing & Publishing",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B01",
    "Catname": "C1300",
    "Catorder": "Commercial printing & typesetting",
    "Industry": "Printing & Publishing",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B01",
    "Catname": "C1400",
    "Catorder": "Greeting card publishing",
    "Industry": "Printing & Publishing",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B02",
    "Catname": "C2000",
    "Catorder": "Entertainment Industry/Broadcast & Motion Pictures",
    "Industry": "TV/Movies/Music",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B02",
    "Catname": "C2100",
    "Catorder": "Commercial TV & radio stations",
    "Industry": "TV/Movies/Music",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B02",
    "Catname": "C2200",
    "Catorder": "Cable & satellite TV production",
    "Industry": "TV/Movies/Music",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B02",
    "Catname": "C2300",
    "Catorder": "TV production",
    "Industry": "TV/Movies/Music",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B02",
    "Catname": "C2400",
    "Catorder": "Motion Picture production & distribution",
    "Industry": "TV/Movies/Music",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B02",
    "Catname": "C2600",
    "Catorder": "Recorded Music & music production",
    "Industry": "TV/Movies/Music",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B02",
    "Catname": "C2700",
    "Catorder": "Movie Theaters",
    "Industry": "TV/Movies/Music",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B02",
    "Catname": "C2800",
    "Catorder": "Bands, orchestras & other live music production",
    "Industry": "TV/Movies/Music",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B02",
    "Catname": "C2900",
    "Catorder": "Live theater & other entertainment productions",
    "Industry": "TV/Movies/Music",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B09",
    "Catname": "C4000",
    "Catorder": "Telecommunications",
    "Industry": "Telecom Services",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B08",
    "Catname": "C4100",
    "Catorder": "Telephone utilities",
    "Industry": "Telephone Utilities",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B08",
    "Catname": "C4200",
    "Catorder": "Long-distance telephone & telegraph service",
    "Industry": "Telephone Utilities",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B09",
    "Catname": "C4300",
    "Catorder": "Cell/wireless service providers",
    "Industry": "Telecom Services",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B09",
    "Catname": "C4400",
    "Catorder": "Satellite communications",
    "Industry": "Telecom Services",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B09",
    "Catname": "C4500",
    "Catorder": "Cable distributors & service providers",
    "Industry": "Telecom Services",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B09",
    "Catname": "C4600",
    "Catorder": "Other Communications Services",
    "Industry": "Telecom Services",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B12",
    "Catname": "C5000",
    "Catorder": "Electronics manufacturing & services",
    "Industry": "Electronics Mfg & Equip",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B12",
    "Catname": "C5100",
    "Catorder": "Computer manufacture & services",
    "Industry": "Electronics Mfg & Equip",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B12",
    "Catname": "C5110",
    "Catorder": "Computer components & accessories",
    "Industry": "Electronics Mfg & Equip",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B12",
    "Catname": "C5120",
    "Catorder": "Computer software",
    "Industry": "Electronics Mfg & Equip",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B12",
    "Catname": "C5130",
    "Catorder": "Data processing & computer services",
    "Industry": "Electronics Mfg & Equip",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B12",
    "Catname": "C5200",
    "Catorder": "Telecommunications Devices",
    "Industry": "Electronics Mfg & Equip",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B12",
    "Catname": "C5300",
    "Catorder": "Other Communication Electronics",
    "Industry": "Electronics Mfg & Equip",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B12",
    "Catname": "C5400",
    "Catorder": "Industrial Electronics",
    "Industry": "Electronics Mfg & Equip",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B13",
    "Catname": "C6000",
    "Catorder": "Internet & Online Services",
    "Industry": "Internet",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B13",
    "Catname": "C6100",
    "Catorder": "Online Entertainment",
    "Industry": "Internet",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B13",
    "Catname": "C6200",
    "Catorder": "Social Media",
    "Industry": "Internet",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B13",
    "Catname": "C6300",
    "Catorder": "Search Engine/Email Services",
    "Industry": "Internet",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B13",
    "Catname": "C6400",
    "Catorder": "Vendors",
    "Industry": "Internet",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "B13",
    "Catname": "C6500",
    "Catorder": "Hosting/Cloud Services",
    "Industry": "Internet",
    "Sector": "Communic/Electronics",
    "Sector Long": "Communications/Electronics"
  },
  {
    "Catcode": "D03",
    "Catname": "D0000",
    "Catorder": "Defense",
    "Industry": "Misc Defense",
    "Sector": "Defense",
    "Sector Long": "Defense"
  },
  {
    "Catcode": "D01",
    "Catname": "D2000",
    "Catorder": "Defense aerospace contractors",
    "Industry": "Defense Aerospace",
    "Sector": "Defense",
    "Sector Long": "Defense"
  },
  {
    "Catcode": "D02",
    "Catname": "D3000",
    "Catorder": "Defense electronic contractors",
    "Industry": "Defense Electronics",
    "Sector": "Defense",
    "Sector Long": "Defense"
  },
  {
    "Catcode": "D03",
    "Catname": "D4000",
    "Catorder": "Defense Research & Development",
    "Industry": "Misc Defense",
    "Sector": "Defense",
    "Sector Long": "Defense"
  },
  {
    "Catcode": "D03",
    "Catname": "D5000",
    "Catorder": "Defense shipbuilders",
    "Industry": "Misc Defense",
    "Sector": "Defense",
    "Sector Long": "Defense"
  },
  {
    "Catcode": "D03",
    "Catname": "D6000",
    "Catorder": "Homeland Security contractors",
    "Industry": "Misc Defense",
    "Sector": "Defense",
    "Sector Long": "Defense"
  },
  {
    "Catcode": "D03",
    "Catname": "D8000",
    "Catorder": "Ground-based & other weapons systems",
    "Industry": "Misc Defense",
    "Sector": "Defense",
    "Sector Long": "Defense"
  },
  {
    "Catcode": "D03",
    "Catname": "D9000",
    "Catorder": "Defense-related services",
    "Industry": "Misc Defense",
    "Sector": "Defense",
    "Sector Long": "Defense"
  },
  {
    "Catcode": "E07",
    "Catname": "E0000",
    "Catorder": "Energy, Natural Resources and Environment",
    "Industry": "Misc Energy",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "E07",
    "Catname": "E1000",
    "Catorder": "Energy production & distribution",
    "Industry": "Misc Energy",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "E01",
    "Catname": "E1100",
    "Catorder": "Oil & Gas",
    "Industry": "Oil & Gas",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "E01",
    "Catname": "E1110",
    "Catorder": "Major (multinational) oil & gas producers",
    "Industry": "Oil & Gas",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "E01",
    "Catname": "E1120",
    "Catorder": "Independent oil & gas producers",
    "Industry": "Oil & Gas",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "E01",
    "Catname": "E1140",
    "Catorder": "Natural Gas transmission & distribution",
    "Industry": "Oil & Gas",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "E01",
    "Catname": "E1150",
    "Catorder": "Oilfield service, equipment & exploration",
    "Industry": "Oil & Gas",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "E01",
    "Catname": "E1160",
    "Catorder": "Petroleum refining & marketing",
    "Industry": "Oil & Gas",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "E01",
    "Catname": "E1170",
    "Catorder": "Gasoline service stations",
    "Industry": "Oil & Gas",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "E01",
    "Catname": "E1180",
    "Catorder": "Fuel oil dealers",
    "Industry": "Oil & Gas",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "E01",
    "Catname": "E1190",
    "Catorder": "LPG/Liquid Propane dealers & producers",
    "Industry": "Oil & Gas",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "E04",
    "Catname": "E1200",
    "Catorder": "Mining",
    "Industry": "Mining",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "E04",
    "Catname": "E1210",
    "Catorder": "Coal mining",
    "Industry": "Mining",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "E04",
    "Catname": "E1220",
    "Catorder": "Metal mining & processing",
    "Industry": "Mining",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "E04",
    "Catname": "E1230",
    "Catorder": "Non-metallic mining",
    "Industry": "Mining",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "E04",
    "Catname": "E1240",
    "Catorder": "Mining services & equipment",
    "Industry": "Mining",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "E08",
    "Catname": "E1300",
    "Catorder": "Nuclear energy",
    "Industry": "Electric Utilities",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "E08",
    "Catname": "E1320",
    "Catorder": "Nuclear plant construction, equipment & svcs",
    "Industry": "Electric Utilities",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "E07",
    "Catname": "E1500",
    "Catorder": "Alternate energy production & services",
    "Industry": "Misc Energy",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "E08",
    "Catname": "E1600",
    "Catorder": "Electric Power utilities",
    "Industry": "Electric Utilities",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "E08",
    "Catname": "E1610",
    "Catorder": "Rural electric cooperatives",
    "Industry": "Electric Utilities",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "E08",
    "Catname": "E1620",
    "Catorder": "Gas & Electric Utilities",
    "Industry": "Electric Utilities",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "E08",
    "Catname": "E1630",
    "Catorder": "Independent power generation & cogeneration",
    "Industry": "Electric Utilities",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "E07",
    "Catname": "E1700",
    "Catorder": "Power plant construction & equipment",
    "Industry": "Misc Energy",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "E09",
    "Catname": "E2000",
    "Catorder": "Environmental services, equipment & consulting",
    "Industry": "Environmental Svcs/Equipment",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "E10",
    "Catname": "E3000",
    "Catorder": "Waste management",
    "Industry": "Waste Management",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "E11",
    "Catname": "E4000",
    "Catorder": "Fisheries & wildlife",
    "Industry": "Fisheries & Wildlife",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "E11",
    "Catname": "E4100",
    "Catorder": "Fishing",
    "Industry": "Fisheries & Wildlife",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "E11",
    "Catname": "E4200",
    "Catorder": "Hunting & wildlife",
    "Industry": "Fisheries & Wildlife",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "E07",
    "Catname": "E5000",
    "Catorder": "Water Utilities",
    "Industry": "Misc Energy",
    "Sector": "Energy/Nat Resource",
    "Sector Long": "Energy & Natural Resources"
  },
  {
    "Catcode": "F13",
    "Catname": "F0000",
    "Catorder": "Finance, Insurance & Real Estate",
    "Industry": "Misc Finance",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F03",
    "Catname": "F1000",
    "Catorder": "Banks & lending institutions",
    "Industry": "Commercial Banks",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F03",
    "Catname": "F1100",
    "Catorder": "Commercial banks & bank holding companies",
    "Industry": "Commercial Banks",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F04",
    "Catname": "F1200",
    "Catorder": "Savings banks & Savings and Loans",
    "Industry": "Savings & Loans",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F05",
    "Catname": "F1300",
    "Catorder": "Credit unions",
    "Industry": "Credit Unions",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F06",
    "Catname": "F1400",
    "Catorder": "Credit agencies & finance companies",
    "Industry": "Finance/Credit Companies",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F06",
    "Catname": "F1410",
    "Catorder": "Student loan companies",
    "Industry": "Finance/Credit Companies",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F06",
    "Catname": "F1420",
    "Catorder": "Payday lenders",
    "Industry": "Finance/Credit Companies",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F07",
    "Catname": "F2000",
    "Catorder": "Securities, commodities & investment",
    "Industry": "Securities & Investment",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F07",
    "Catname": "F2100",
    "Catorder": "Security brokers & investment companies",
    "Industry": "Securities & Investment",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F07",
    "Catname": "F2110",
    "Catorder": "Discount & Online Brokers",
    "Industry": "Securities & Investment",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F07",
    "Catname": "F2200",
    "Catorder": "Commodity brokers/dealers",
    "Industry": "Securities & Investment",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F07",
    "Catname": "F2300",
    "Catorder": "Investment banking",
    "Industry": "Securities & Investment",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F07",
    "Catname": "F2400",
    "Catorder": "Stock exchanges",
    "Industry": "Securities & Investment",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F07",
    "Catname": "F2500",
    "Catorder": "Venture capital",
    "Industry": "Securities & Investment",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F07",
    "Catname": "F2600",
    "Catorder": "Private Equity & Investment Firms",
    "Industry": "Securities & Investment",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F07",
    "Catname": "F2700",
    "Catorder": "Hedge Funds",
    "Industry": "Securities & Investment",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F09",
    "Catname": "F3000",
    "Catorder": "Insurance",
    "Industry": "Insurance",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F09",
    "Catname": "F3100",
    "Catorder": "Insurance companies, brokers & agents",
    "Industry": "Insurance",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F09",
    "Catname": "F3200",
    "Catorder": "Accident & health insurance",
    "Industry": "Insurance",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F09",
    "Catname": "F3300",
    "Catorder": "Life insurance",
    "Industry": "Insurance",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F09",
    "Catname": "F3400",
    "Catorder": "Property & casualty insurance",
    "Industry": "Insurance",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F10",
    "Catname": "F4000",
    "Catorder": "Real estate",
    "Industry": "Real Estate",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F10",
    "Catname": "F4100",
    "Catorder": "Real Estate developers & subdividers",
    "Industry": "Real Estate",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F10",
    "Catname": "F4200",
    "Catorder": "Real estate agents",
    "Industry": "Real Estate",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F10",
    "Catname": "F4300",
    "Catorder": "Title insurance & title abstract offices",
    "Industry": "Real Estate",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F10",
    "Catname": "F4400",
    "Catorder": "Mobile home dealers & parks",
    "Industry": "Real Estate",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F10",
    "Catname": "F4500",
    "Catorder": "Building operators and managers",
    "Industry": "Real Estate",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F10",
    "Catname": "F4600",
    "Catorder": "Mortgage bankers and brokers",
    "Industry": "Real Estate",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F10",
    "Catname": "F4700",
    "Catorder": "Other real estate services",
    "Industry": "Real Estate",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F13",
    "Catname": "F5000",
    "Catorder": "Financial services & consulting",
    "Industry": "Misc Finance",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F11",
    "Catname": "F5100",
    "Catorder": "Accountants",
    "Industry": "Accountants",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F13",
    "Catname": "F5200",
    "Catorder": "Credit reporting services & collection agencies",
    "Industry": "Misc Finance",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F13",
    "Catname": "F5300",
    "Catorder": "Tax return services",
    "Industry": "Misc Finance",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F13",
    "Catname": "F5500",
    "Catorder": "Other financial services",
    "Industry": "Misc Finance",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "F13",
    "Catname": "F7000",
    "Catorder": "Investors",
    "Industry": "Misc Finance",
    "Sector": "Finance/Insur/RealEst",
    "Sector Long": "Finance, Insurance & Real Estate"
  },
  {
    "Catcode": "N12",
    "Catname": "G0000",
    "Catorder": "General commerce",
    "Industry": "Misc Business",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N00",
    "Catname": "G1000",
    "Catorder": "General business associations",
    "Industry": "Business Associations",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N00",
    "Catname": "G1100",
    "Catorder": "Chambers of commerce",
    "Industry": "Business Associations",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N00",
    "Catname": "G1200",
    "Catorder": "Small business associations",
    "Industry": "Business Associations",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N00",
    "Catname": "G1300",
    "Catorder": "Pro-business associations",
    "Industry": "Business Associations",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N00",
    "Catname": "G1310",
    "Catorder": "Business tax coalitions",
    "Industry": "Business Associations",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N00",
    "Catname": "G1400",
    "Catorder": "International trade associations",
    "Industry": "Business Associations",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "A09",
    "Catname": "G2000",
    "Catorder": "Food & Beverage Products and Services",
    "Industry": "Food Processing & Sales",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "A09",
    "Catname": "G2100",
    "Catorder": "Food and kindred products manufacturing",
    "Industry": "Food Processing & Sales",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "N01",
    "Catname": "G2110",
    "Catorder": "Artificial sweeteners and food additives",
    "Industry": "Food & Beverage",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N01",
    "Catname": "G2200",
    "Catorder": "Confectionery processors & manufacturers",
    "Industry": "Food & Beverage",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "A09",
    "Catname": "G2300",
    "Catorder": "Meat processing & products",
    "Industry": "Food Processing & Sales",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "N01",
    "Catname": "G2350",
    "Catorder": "Fish Processing",
    "Industry": "Food & Beverage",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "A09",
    "Catname": "G2400",
    "Catorder": "Food stores",
    "Industry": "Food Processing & Sales",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "A09",
    "Catname": "G2500",
    "Catorder": "Food wholesalers",
    "Industry": "Food Processing & Sales",
    "Sector": "Agribusiness",
    "Sector Long": "Agribusiness"
  },
  {
    "Catcode": "N01",
    "Catname": "G2600",
    "Catorder": "Beverages (non-alcoholic)",
    "Industry": "Food & Beverage",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N01",
    "Catname": "G2700",
    "Catorder": "Beverage bottling & distribution",
    "Industry": "Food & Beverage",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N02",
    "Catname": "G2800",
    "Catorder": "Alcohol",
    "Industry": "Beer, Wine & Liquor",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N02",
    "Catname": "G2810",
    "Catorder": "Beer",
    "Industry": "Beer, Wine & Liquor",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N02",
    "Catname": "G2820",
    "Catorder": "Wine & distilled spirits manufacturing",
    "Industry": "Beer, Wine & Liquor",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N02",
    "Catname": "G2840",
    "Catorder": "Liquor stores",
    "Industry": "Beer, Wine & Liquor",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N02",
    "Catname": "G2850",
    "Catorder": "Liquor wholesalers",
    "Industry": "Beer, Wine & Liquor",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N09",
    "Catname": "G2860",
    "Catorder": "Marijuana Production, Sales & Paraphernalia",
    "Industry": "Marijuana",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N01",
    "Catname": "G2900",
    "Catorder": "Restaurants & drinking establishments",
    "Industry": "Food & Beverage",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N01",
    "Catname": "G2910",
    "Catorder": "Food catering & food services",
    "Industry": "Food & Beverage",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N12",
    "Catname": "G3000",
    "Catorder": "Wholesale trade",
    "Industry": "Misc Business",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N12",
    "Catname": "G3500",
    "Catorder": "Import/Export services",
    "Industry": "Misc Business",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N03",
    "Catname": "G4000",
    "Catorder": "Retail trade",
    "Industry": "Retail Sales",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N03",
    "Catname": "G4100",
    "Catorder": "Apparel & accessory stores",
    "Industry": "Retail Sales",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N03",
    "Catname": "G4200",
    "Catorder": "Consumer electronics & computer stores",
    "Industry": "Retail Sales",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N03",
    "Catname": "G4300",
    "Catorder": "Department, variety & convenience stores",
    "Industry": "Retail Sales",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N03",
    "Catname": "G4400",
    "Catorder": "Furniture & appliance stores",
    "Industry": "Retail Sales",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N03",
    "Catname": "G4500",
    "Catorder": "Hardware & building materials stores",
    "Industry": "Retail Sales",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N03",
    "Catname": "G4600",
    "Catorder": "Miscellaneous retail stores",
    "Industry": "Retail Sales",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N03",
    "Catname": "G4700",
    "Catorder": "Catalog & mail order houses",
    "Industry": "Retail Sales",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N03",
    "Catname": "G4800",
    "Catorder": "Direct sales",
    "Industry": "Retail Sales",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N03",
    "Catname": "G4850",
    "Catorder": "Vending Machine Sales & Services",
    "Industry": "Retail Sales",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N03",
    "Catname": "G4900",
    "Catorder": "Drug stores",
    "Industry": "Retail Sales",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N04",
    "Catname": "G5000",
    "Catorder": "Services",
    "Industry": "Misc Services",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N04",
    "Catname": "G5100",
    "Catorder": "Beauty & barber shops",
    "Industry": "Misc Services",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N05",
    "Catname": "G5200",
    "Catorder": "Business services",
    "Industry": "Business Services",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N05",
    "Catname": "G5210",
    "Catorder": "Advertising & public relations services",
    "Industry": "Business Services",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N05",
    "Catname": "G5220",
    "Catorder": "Direct mail advertising services",
    "Industry": "Business Services",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N05",
    "Catname": "G5230",
    "Catorder": "Outdoor advertising services",
    "Industry": "Business Services",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N05",
    "Catname": "G5240",
    "Catorder": "Commercial photography, art & graphic design",
    "Industry": "Business Services",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N05",
    "Catname": "G5250",
    "Catorder": "Employment agencies",
    "Industry": "Business Services",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N05",
    "Catname": "G5260",
    "Catorder": "Political consultants/advisers",
    "Industry": "Business Services",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N05",
    "Catname": "G5270",
    "Catorder": "Management consultants & services",
    "Industry": "Business Services",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N05",
    "Catname": "G5280",
    "Catorder": "Marketing research services",
    "Industry": "Business Services",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N05",
    "Catname": "G5290",
    "Catorder": "Security services",
    "Industry": "Business Services",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N04",
    "Catname": "G5300",
    "Catorder": "Equipment rental & leasing",
    "Industry": "Misc Services",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N04",
    "Catname": "G5400",
    "Catorder": "Funeral services",
    "Industry": "Misc Services",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N04",
    "Catname": "G5500",
    "Catorder": "Laundries & dry cleaners",
    "Industry": "Misc Services",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N04",
    "Catname": "G5600",
    "Catorder": "Miscellaneous repair services",
    "Industry": "Misc Services",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N04",
    "Catname": "G5700",
    "Catorder": "Pest control",
    "Industry": "Misc Services",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N04",
    "Catname": "G5800",
    "Catorder": "Physical fitness centers",
    "Industry": "Misc Services",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N06",
    "Catname": "G6000",
    "Catorder": "Recreation/Entertainment",
    "Industry": "Recreation/Live Entertainment",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N06",
    "Catname": "G6100",
    "Catorder": "Amusement/recreation centers",
    "Industry": "Recreation/Live Entertainment",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N06",
    "Catname": "G6400",
    "Catorder": "Professional sports, arenas & related equip & svcs",
    "Industry": "Recreation/Live Entertainment",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N07",
    "Catname": "G6500",
    "Catorder": "Casinos, racetracks & gambling",
    "Industry": "Casinos/Gambling",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N07",
    "Catname": "G6550",
    "Catorder": "Indian Gaming",
    "Industry": "Casinos/Gambling",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N06",
    "Catname": "G6700",
    "Catorder": "Amusement parks",
    "Industry": "Recreation/Live Entertainment",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N04",
    "Catname": "G6800",
    "Catorder": "Video rental",
    "Industry": "Misc Services",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N12",
    "Catname": "G7000",
    "Catorder": "Correctional facilities constr & mgmt/for-profit",
    "Industry": "Misc Business",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "H05",
    "Catname": "H0000",
    "Catorder": "Health, Education & Human Resources",
    "Industry": "Misc Health",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H01",
    "Catname": "H1000",
    "Catorder": "Health professionals",
    "Industry": "Health Professionals",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H01",
    "Catname": "H1100",
    "Catorder": "Physicians",
    "Industry": "Health Professionals",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H01",
    "Catname": "H1110",
    "Catorder": "Psychiatrists & psychologists",
    "Industry": "Health Professionals",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H01",
    "Catname": "H1120",
    "Catorder": "Optometrists & Ophthalmologists",
    "Industry": "Health Professionals",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H01",
    "Catname": "H1130",
    "Catorder": "Other physician specialists",
    "Industry": "Health Professionals",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H01",
    "Catname": "H1400",
    "Catorder": "Dentists",
    "Industry": "Health Professionals",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H01",
    "Catname": "H1500",
    "Catorder": "Chiropractors",
    "Industry": "Health Professionals",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H01",
    "Catname": "H1700",
    "Catorder": "Other non-physician health practitioners",
    "Industry": "Health Professionals",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H01",
    "Catname": "H1710",
    "Catorder": "Nurses",
    "Industry": "Health Professionals",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H01",
    "Catname": "H1750",
    "Catorder": "Pharmacists",
    "Industry": "Health Professionals",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H02",
    "Catname": "H2000",
    "Catorder": "Health care institutions",
    "Industry": "Hospitals/Nursing Homes",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H02",
    "Catname": "H2100",
    "Catorder": "Hospitals",
    "Industry": "Hospitals/Nursing Homes",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H02",
    "Catname": "H2200",
    "Catorder": "Nursing homes",
    "Industry": "Hospitals/Nursing Homes",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H02",
    "Catname": "H2300",
    "Catorder": "Drug & alcohol treatment hospitals",
    "Industry": "Hospitals/Nursing Homes",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H03",
    "Catname": "H3000",
    "Catorder": "Health care services",
    "Industry": "Health Services/HMOs",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H03",
    "Catname": "H3100",
    "Catorder": "Home care services",
    "Industry": "Health Services/HMOs",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H03",
    "Catname": "H3200",
    "Catorder": "Outpatient health services (incl drug & alcohol)",
    "Industry": "Health Services/HMOs",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H03",
    "Catname": "H3300",
    "Catorder": "Optical services (glasses & contact lenses)",
    "Industry": "Health Services/HMOs",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H03",
    "Catname": "H3400",
    "Catorder": "Medical laboratories",
    "Industry": "Health Services/HMOs",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H03",
    "Catname": "H3500",
    "Catorder": "AIDS treatment & testing",
    "Industry": "Health Services/HMOs",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H03",
    "Catname": "H3700",
    "Catorder": "HMOs",
    "Industry": "Health Services/HMOs",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H03",
    "Catname": "H3800",
    "Catorder": "Mental Health Services",
    "Industry": "Health Services/HMOs",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H03",
    "Catname": "H3900",
    "Catorder": "Health care Consultants",
    "Industry": "Health Services/HMOs",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H04",
    "Catname": "H4000",
    "Catorder": "Health care products",
    "Industry": "Pharmaceuticals/Health Products",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H04",
    "Catname": "H4100",
    "Catorder": "Medical Devices & Supplies",
    "Industry": "Pharmaceuticals/Health Products",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H04",
    "Catname": "H4200",
    "Catorder": "Personal health care products",
    "Industry": "Pharmaceuticals/Health Products",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H04",
    "Catname": "H4300",
    "Catorder": "Pharmaceutical manufacturing",
    "Industry": "Pharmaceuticals/Health Products",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H04",
    "Catname": "H4400",
    "Catorder": "Pharmaceutical wholesale",
    "Industry": "Pharmaceuticals/Health Products",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H04",
    "Catname": "H4500",
    "Catorder": "Biotech products & research",
    "Industry": "Pharmaceuticals/Health Products",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H04",
    "Catname": "H4600",
    "Catorder": "Nutritional & dietary supplements",
    "Industry": "Pharmaceuticals/Health Products",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "H04",
    "Catname": "H4700",
    "Catorder": "Pharmaceutical cannabis",
    "Industry": "Pharmaceuticals/Health Products",
    "Sector": "Health",
    "Sector Long": "Health"
  },
  {
    "Catcode": "W04",
    "Catname": "H5000",
    "Catorder": "Education",
    "Industry": "Education",
    "Sector": "Other",
    "Sector Long": "Other"
  },
  {
    "Catcode": "W04",
    "Catname": "H5100",
    "Catorder": "Schools & colleges",
    "Industry": "Education",
    "Sector": "Other",
    "Sector Long": "Other"
  },
  {
    "Catcode": "W04",
    "Catname": "H5150",
    "Catorder": "Medical schools",
    "Industry": "Education",
    "Sector": "Other",
    "Sector Long": "Other"
  },
  {
    "Catcode": "W04",
    "Catname": "H5170",
    "Catorder": "Law schools",
    "Industry": "Education",
    "Sector": "Other",
    "Sector Long": "Other"
  },
  {
    "Catcode": "W04",
    "Catname": "H5200",
    "Catorder": "Technical, business and vocational schools & svcs",
    "Industry": "Education",
    "Sector": "Other",
    "Sector Long": "Other"
  },
  {
    "Catcode": "W04",
    "Catname": "H5300",
    "Catorder": "For-profit Education",
    "Industry": "Education",
    "Sector": "Other",
    "Sector Long": "Other"
  },
  {
    "Catcode": "W07",
    "Catname": "H6000",
    "Catorder": "Welfare & Social Work",
    "Industry": "Other",
    "Sector": "Other",
    "Sector Long": "Other"
  },
  {
    "Catcode": "Q10",
    "Catname": "J0000",
    "Catorder": "Ideological & Single Issue PACs",
    "Industry": "Misc Issues",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q10",
    "Catname": "J1000",
    "Catorder": "General Ideological",
    "Industry": "Misc Issues",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q01",
    "Catname": "J1100",
    "Catorder": "Republican/Conservative",
    "Industry": "Republican/Conservative",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q01",
    "Catname": "J1110",
    "Catorder": "Christian Conservative",
    "Industry": "Republican/Conservative",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q02",
    "Catname": "J1200",
    "Catorder": "Democratic/Liberal",
    "Industry": "Democratic/Liberal",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q10",
    "Catname": "J1300",
    "Catorder": "Third-party committees",
    "Industry": "Misc Issues",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q03",
    "Catname": "J2000",
    "Catorder": "Leadership committees",
    "Industry": "Leadership PACs",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q03",
    "Catname": "J2100",
    "Catorder": "Democratic leadership PAC",
    "Industry": "Leadership PACs",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q03",
    "Catname": "J2200",
    "Catorder": "Republican leadership PAC",
    "Industry": "Leadership PACs",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q03",
    "Catname": "J2300",
    "Catorder": "Democratic officials, candidates & former members",
    "Industry": "Leadership PACs",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q03",
    "Catname": "J2400",
    "Catorder": "Republican officials, candidates & former members",
    "Industry": "Leadership PACs",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q03",
    "Catname": "J2500",
    "Catorder": "Non-Federal candidate committees",
    "Industry": "Leadership PACs",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q10",
    "Catname": "J3000",
    "Catorder": "Consumer groups",
    "Industry": "Misc Issues",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q10",
    "Catname": "J4000",
    "Catorder": "Fiscal & tax policy",
    "Industry": "Misc Issues",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q04",
    "Catname": "J5000",
    "Catorder": "Foreign policy",
    "Industry": "Foreign & Defense Policy",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q05",
    "Catname": "J5100",
    "Catorder": "Pro-Israel",
    "Industry": "Pro-Israel",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q04",
    "Catname": "J5200",
    "Catorder": "Anti-Castro",
    "Industry": "Foreign & Defense Policy",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q04",
    "Catname": "J5300",
    "Catorder": "Puerto Rico statehood policy",
    "Industry": "Foreign & Defense Policy",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q04",
    "Catname": "J5400",
    "Catorder": "Pro-Arab",
    "Industry": "Foreign & Defense Policy",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q12",
    "Catname": "J6100",
    "Catorder": "Anti-Guns",
    "Industry": "Gun Control",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q13",
    "Catname": "J6200",
    "Catorder": "Pro-Guns",
    "Industry": "Gun Rights",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q10",
    "Catname": "J6500",
    "Catorder": "Militias & Anti-Government Groups",
    "Industry": "Misc Issues",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q09",
    "Catname": "J7000",
    "Catorder": "Human Rights",
    "Industry": "Human Rights",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q14",
    "Catname": "J7120",
    "Catorder": "Abortion policy/Anti-Abortion",
    "Industry": "Abortion Policy/Anti-Abortion",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q15",
    "Catname": "J7150",
    "Catorder": "Abortion policy/Pro-Abortion Rights",
    "Industry": "Abortion Policy/Pro-Abortion Rights",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q10",
    "Catname": "J7200",
    "Catorder": "Elderly issues/Social Security",
    "Industry": "Misc Issues",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q10",
    "Catname": "J7210",
    "Catorder": "Legalization of Doctor-Assisted Suicide",
    "Industry": "Misc Issues",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q09",
    "Catname": "J7300",
    "Catorder": "Gay & lesbian rights & issues",
    "Industry": "Human Rights",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q08",
    "Catname": "J7400",
    "Catorder": "Women's issues",
    "Industry": "Women's Issues",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q09",
    "Catname": "J7500",
    "Catorder": "Minority/Ethnic Groups",
    "Industry": "Human Rights",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q09",
    "Catname": "J7510",
    "Catorder": "Native American Tribes",
    "Industry": "Human Rights",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q10",
    "Catname": "J7600",
    "Catorder": "Animal Rights",
    "Industry": "Misc Issues",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q09",
    "Catname": "J7700",
    "Catorder": "Children's rights",
    "Industry": "Human Rights",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q10",
    "Catname": "J8000",
    "Catorder": "Labor, anti-union",
    "Industry": "Misc Issues",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q10",
    "Catname": "J9000",
    "Catorder": "Other single-issue or ideological groups",
    "Industry": "Misc Issues",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q10",
    "Catname": "J9100",
    "Catorder": "Term limits",
    "Industry": "Misc Issues",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q04",
    "Catname": "JD100",
    "Catorder": "Defense policy, hawks",
    "Industry": "Foreign & Defense Policy",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q04",
    "Catname": "JD200",
    "Catorder": "Defense policy, doves",
    "Industry": "Foreign & Defense Policy",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q11",
    "Catname": "JE300",
    "Catorder": "Environmental policy",
    "Industry": "Environment",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q09",
    "Catname": "JH100",
    "Catorder": "Health & welfare policy",
    "Industry": "Human Rights",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q10",
    "Catname": "JW100",
    "Catorder": "Pro-resource development groups",
    "Industry": "Misc Issues",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "K01",
    "Catname": "K0000",
    "Catorder": "Legal Services",
    "Industry": "Lawyers/Law Firms",
    "Sector": "Lawyers & Lobbyists",
    "Sector Long": "Lawyers & Lobbyists"
  },
  {
    "Catcode": "K01",
    "Catname": "K1000",
    "Catorder": "Attorneys & law firms",
    "Industry": "Lawyers/Law Firms",
    "Sector": "Lawyers & Lobbyists",
    "Sector Long": "Lawyers & Lobbyists"
  },
  {
    "Catcode": "K01",
    "Catname": "K1100",
    "Catorder": "Trial lawyers & law firms",
    "Industry": "Lawyers/Law Firms",
    "Sector": "Lawyers & Lobbyists",
    "Sector Long": "Lawyers & Lobbyists"
  },
  {
    "Catcode": "K01",
    "Catname": "K1200",
    "Catorder": "Corporate lawyers & law firms",
    "Industry": "Lawyers/Law Firms",
    "Sector": "Lawyers & Lobbyists",
    "Sector Long": "Lawyers & Lobbyists"
  },
  {
    "Catcode": "K02",
    "Catname": "K2000",
    "Catorder": "Lobbyists & Public Relations",
    "Industry": "Lobbyists",
    "Sector": "Lawyers & Lobbyists",
    "Sector Long": "Lawyers & Lobbyists"
  },
  {
    "Catcode": "K02",
    "Catname": "K2100",
    "Catorder": "Registered Foreign Agents",
    "Industry": "Lobbyists",
    "Sector": "Lawyers & Lobbyists",
    "Sector Long": "Lawyers & Lobbyists"
  },
  {
    "Catcode": "P05",
    "Catname": "L0000",
    "Catorder": "Labor Unions",
    "Industry": "Misc Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "P04",
    "Catname": "L1000",
    "Catorder": "Civil service & government unions",
    "Industry": "Public Sector Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "P04",
    "Catname": "L1100",
    "Catorder": "Federal employees unions",
    "Industry": "Public Sector Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "P04",
    "Catname": "L1200",
    "Catorder": "State & local govt employee unions",
    "Industry": "Public Sector Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "P04",
    "Catname": "L1300",
    "Catorder": "Teachers unions",
    "Industry": "Public Sector Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "P04",
    "Catname": "L1400",
    "Catorder": "Police & firefighters unions & associations",
    "Industry": "Public Sector Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "P04",
    "Catname": "L1500",
    "Catorder": "US Postal Service unions & associations",
    "Industry": "Public Sector Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "P05",
    "Catname": "L5000",
    "Catorder": "Other unions",
    "Industry": "Misc Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "P05",
    "Catname": "LA100",
    "Catorder": "Agricultural labor unions",
    "Industry": "Misc Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "P01",
    "Catname": "LB100",
    "Catorder": "Building trades unions",
    "Industry": "Building Trade Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "P02",
    "Catname": "LC100",
    "Catorder": "Communications & hi-tech unions",
    "Industry": "Industrial Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "P02",
    "Catname": "LC150",
    "Catorder": "IBEW (Intl Brotherhood of Electrical Workers)",
    "Industry": "Industrial Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "P05",
    "Catname": "LD100",
    "Catorder": "Defense-related unions",
    "Industry": "Misc Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "P02",
    "Catname": "LE100",
    "Catorder": "Mining unions",
    "Industry": "Industrial Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "P02",
    "Catname": "LE200",
    "Catorder": "Energy-related unions (non-mining)",
    "Industry": "Industrial Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "P05",
    "Catname": "LG000",
    "Catorder": "General commercial unions",
    "Industry": "Misc Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "P05",
    "Catname": "LG100",
    "Catorder": "Food service & related unions",
    "Industry": "Misc Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "P05",
    "Catname": "LG200",
    "Catorder": "Retail trade unions",
    "Industry": "Misc Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "P05",
    "Catname": "LG300",
    "Catorder": "Commercial service unions",
    "Industry": "Misc Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "P05",
    "Catname": "LG400",
    "Catorder": "Entertainment unions",
    "Industry": "Misc Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "P05",
    "Catname": "LG500",
    "Catorder": "Other commercial unions",
    "Industry": "Misc Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "P05",
    "Catname": "LH100",
    "Catorder": "Health worker unions",
    "Industry": "Misc Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "P02",
    "Catname": "LM100",
    "Catorder": "Manufacturing unions",
    "Industry": "Industrial Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "P02",
    "Catname": "LM150",
    "Catorder": "Automotive Manufacturing union",
    "Industry": "Industrial Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "P03",
    "Catname": "LT000",
    "Catorder": "Transportation unions",
    "Industry": "Transportation Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "P03",
    "Catname": "LT100",
    "Catorder": "Air transport unions",
    "Industry": "Transportation Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "P03",
    "Catname": "LT300",
    "Catorder": "Teamsters union",
    "Industry": "Transportation Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "P03",
    "Catname": "LT400",
    "Catorder": "Railroad unions",
    "Industry": "Transportation Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "P03",
    "Catname": "LT500",
    "Catorder": "Merchant marine & longshoremen unions",
    "Industry": "Transportation Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "P03",
    "Catname": "LT600",
    "Catorder": "Other transportation unions",
    "Industry": "Transportation Unions",
    "Sector": "Labor",
    "Sector Long": "Labor"
  },
  {
    "Catcode": "N15",
    "Catname": "M0000",
    "Catorder": "Manufacturing",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N13",
    "Catname": "M1000",
    "Catorder": "Chemicals",
    "Industry": "Chemical & Related Manufacturing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N13",
    "Catname": "M1100",
    "Catorder": "Explosives",
    "Industry": "Chemical & Related Manufacturing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N13",
    "Catname": "M1300",
    "Catorder": "Household cleansers & chemicals",
    "Industry": "Chemical & Related Manufacturing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M1400",
    "Catorder": "Manmade fibers",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N13",
    "Catname": "M1500",
    "Catorder": "Plastics & Rubber processing & products",
    "Industry": "Chemical & Related Manufacturing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N13",
    "Catname": "M1600",
    "Catorder": "Paints, Solvents & Coatings",
    "Industry": "Chemical & Related Manufacturing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N13",
    "Catname": "M1700",
    "Catorder": "Adhesives & Sealants",
    "Industry": "Chemical & Related Manufacturing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M2000",
    "Catorder": "Heavy industrial manufacturing",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N14",
    "Catname": "M2100",
    "Catorder": "Steel",
    "Industry": "Steel Production",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M2200",
    "Catorder": "Smelting & non-petroleum refining",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M2250",
    "Catorder": "Aluminum mining/processing",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M2300",
    "Catorder": "Industrial/commercial equipment & materials",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M2400",
    "Catorder": "Recycling of metal, paper, plastics, etc.",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M3000",
    "Catorder": "Personal products manufacturing",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M3100",
    "Catorder": "Clothing & accessories",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M3200",
    "Catorder": "Shoes & leather products",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M3300",
    "Catorder": "Toiletries & cosmetics",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M3400",
    "Catorder": "Jewelry",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M3500",
    "Catorder": "Toys",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M3600",
    "Catorder": "Sporting goods sales & manufacturing",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M4000",
    "Catorder": "Household & office products",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M4100",
    "Catorder": "Furniture & wood products",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M4200",
    "Catorder": "Office machines",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M4300",
    "Catorder": "Household appliances",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M5000",
    "Catorder": "Fabricated metal products",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M5100",
    "Catorder": "Hardware & tools",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M5200",
    "Catorder": "Electroplating, polishing & related services",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M5300",
    "Catorder": "Small arms & ammunition",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M6000",
    "Catorder": "Electrical lighting products",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M7000",
    "Catorder": "Paper, glass & packaging materials",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M7100",
    "Catorder": "Paper packaging materials",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M7200",
    "Catorder": "Glass products",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M7300",
    "Catorder": "Metal cans & containers",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N16",
    "Catname": "M8000",
    "Catorder": "Textiles & fabrics",
    "Industry": "Textiles",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M9000",
    "Catorder": "Precision instruments",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M9100",
    "Catorder": "Optical instruments & lenses",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M9200",
    "Catorder": "Photographic equipment & supplies",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N15",
    "Catname": "M9300",
    "Catorder": "Clocks & watches",
    "Industry": "Misc Manufacturing & Distributing",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "M06",
    "Catname": "T0000",
    "Catorder": "Transportation",
    "Industry": "Misc Transport",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M01",
    "Catname": "T1000",
    "Catorder": "Air transport",
    "Industry": "Air Transport",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M01",
    "Catname": "T1100",
    "Catorder": "Airlines",
    "Industry": "Air Transport",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M01",
    "Catname": "T1200",
    "Catorder": "Aircraft manufacturers",
    "Industry": "Air Transport",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M01",
    "Catname": "T1300",
    "Catorder": "Aircraft parts & equipment",
    "Industry": "Air Transport",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M01",
    "Catname": "T1400",
    "Catorder": "General aviation (private pilots)",
    "Industry": "Air Transport",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M01",
    "Catname": "T1500",
    "Catorder": "Air freight",
    "Industry": "Air Transport",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M01",
    "Catname": "T1600",
    "Catorder": "Aviation services & airports",
    "Industry": "Air Transport",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M01",
    "Catname": "T1700",
    "Catorder": "Space vehicles & components",
    "Industry": "Air Transport",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M02",
    "Catname": "T2000",
    "Catorder": "Automotive, Misc",
    "Industry": "Automotive",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M02",
    "Catname": "T2100",
    "Catorder": "Auto manufacturers",
    "Industry": "Automotive",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M02",
    "Catname": "T2200",
    "Catorder": "Truck/Automotive parts & accessories",
    "Industry": "Automotive",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M02",
    "Catname": "T2300",
    "Catorder": "Auto dealers, new & used",
    "Industry": "Automotive",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M02",
    "Catname": "T2310",
    "Catorder": "Auto dealers, foreign imports",
    "Industry": "Automotive",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M02",
    "Catname": "T2400",
    "Catorder": "Auto repair",
    "Industry": "Automotive",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M02",
    "Catname": "T2500",
    "Catorder": "Car rental agencies",
    "Industry": "Automotive",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M03",
    "Catname": "T3000",
    "Catorder": "Trucking",
    "Industry": "Trucking",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M03",
    "Catname": "T3100",
    "Catorder": "Trucking companies & services",
    "Industry": "Trucking",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M03",
    "Catname": "T3200",
    "Catorder": "Truck & trailer manufacturers",
    "Industry": "Trucking",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M06",
    "Catname": "T4000",
    "Catorder": "Buses & Taxis",
    "Industry": "Misc Transport",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M06",
    "Catname": "T4100",
    "Catorder": "Bus services",
    "Industry": "Misc Transport",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M06",
    "Catname": "T4200",
    "Catorder": "Taxicabs",
    "Industry": "Misc Transport",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M04",
    "Catname": "T5000",
    "Catorder": "Railroad transportation",
    "Industry": "Railroads",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M04",
    "Catname": "T5100",
    "Catorder": "Railroads",
    "Industry": "Railroads",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M04",
    "Catname": "T5200",
    "Catorder": "Manufacturers of railroad equipment",
    "Industry": "Railroads",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M04",
    "Catname": "T5300",
    "Catorder": "Railroad services",
    "Industry": "Railroads",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M05",
    "Catname": "T6000",
    "Catorder": "Sea transport",
    "Industry": "Sea Transport",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M05",
    "Catname": "T6100",
    "Catorder": "Ship building & repair",
    "Industry": "Sea Transport",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M05",
    "Catname": "T6200",
    "Catorder": "Sea freight & passenger services",
    "Industry": "Sea Transport",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M05",
    "Catname": "T6250",
    "Catorder": "Cruise ships & lines",
    "Industry": "Sea Transport",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M06",
    "Catname": "T7000",
    "Catorder": "Freight & delivery services",
    "Industry": "Misc Transport",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M01",
    "Catname": "T7100",
    "Catorder": "Express delivery services",
    "Industry": "Air Transport",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "N12",
    "Catname": "T7200",
    "Catorder": "Warehousing",
    "Industry": "Misc Business",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "M06",
    "Catname": "T8000",
    "Catorder": "Recreational transport",
    "Industry": "Misc Transport",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M06",
    "Catname": "T8100",
    "Catorder": "Motorcycles, snowmobiles & other motorized vehicle",
    "Industry": "Misc Transport",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M06",
    "Catname": "T8200",
    "Catorder": "Motor homes & camper trailers",
    "Industry": "Misc Transport",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M06",
    "Catname": "T8300",
    "Catorder": "Pleasure boats",
    "Industry": "Misc Transport",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "M06",
    "Catname": "T8400",
    "Catorder": "Bicycles & other non-motorized recreational transp",
    "Industry": "Misc Transport",
    "Sector": "Transportation",
    "Sector Long": "Transportation"
  },
  {
    "Catcode": "N08",
    "Catname": "T9000",
    "Catorder": "Lodging & tourism",
    "Industry": "Lodging/Tourism",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N08",
    "Catname": "T9100",
    "Catorder": "Hotels & motels",
    "Industry": "Lodging/Tourism",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N08",
    "Catname": "T9300",
    "Catorder": "Resorts",
    "Industry": "Lodging/Tourism",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "N08",
    "Catname": "T9400",
    "Catorder": "Travel agents",
    "Industry": "Lodging/Tourism",
    "Sector": "Misc Business",
    "Sector Long": "Misc Business"
  },
  {
    "Catcode": "W07",
    "Catname": "X0000",
    "Catorder": "Other",
    "Industry": "Other",
    "Sector": "Other",
    "Sector Long": "Other"
  },
  {
    "Catcode": "W06",
    "Catname": "X1200",
    "Catorder": "Retired",
    "Industry": "Retired",
    "Sector": "Other",
    "Sector Long": "Other"
  },
  {
    "Catcode": "W03",
    "Catname": "X3000",
    "Catorder": "Civil servant/public employee",
    "Industry": "Civil Servants/Public Officials",
    "Sector": "Other",
    "Sector Long": "Other"
  },
  {
    "Catcode": "W03",
    "Catname": "X3100",
    "Catorder": "Public official (elected or appointed)",
    "Industry": "Civil Servants/Public Officials",
    "Sector": "Other",
    "Sector Long": "Other"
  },
  {
    "Catcode": "W03",
    "Catname": "X3200",
    "Catorder": "Courts & Justice System",
    "Industry": "Civil Servants/Public Officials",
    "Sector": "Other",
    "Sector Long": "Other"
  },
  {
    "Catcode": "W03",
    "Catname": "X3300",
    "Catorder": "Municipal & county government organizations",
    "Industry": "Civil Servants/Public Officials",
    "Sector": "Other",
    "Sector Long": "Other"
  },
  {
    "Catcode": "W04",
    "Catname": "X3500",
    "Catorder": "Public school teachers, administrators & officials",
    "Industry": "Education",
    "Sector": "Other",
    "Sector Long": "Other"
  },
  {
    "Catcode": "W03",
    "Catname": "X3700",
    "Catorder": "US Postal Service",
    "Industry": "Civil Servants/Public Officials",
    "Sector": "Other",
    "Sector Long": "Other"
  },
  {
    "Catcode": "W02",
    "Catname": "X4000",
    "Catorder": "Non-Profits",
    "Industry": "Non-Profit Institutions",
    "Sector": "Other",
    "Sector Long": "Other"
  },
  {
    "Catcode": "W02",
    "Catname": "X4100",
    "Catorder": "Non-profit foundations",
    "Industry": "Non-Profit Institutions",
    "Sector": "Other",
    "Sector Long": "Other"
  },
  {
    "Catcode": "W02",
    "Catname": "X4110",
    "Catorder": "Philanthropists",
    "Industry": "Non-Profit Institutions",
    "Sector": "Other",
    "Sector Long": "Other"
  },
  {
    "Catcode": "W02",
    "Catname": "X4200",
    "Catorder": "Museums, art galleries, libraries, etc.",
    "Industry": "Non-Profit Institutions",
    "Sector": "Other",
    "Sector Long": "Other"
  },
  {
    "Catcode": "W07",
    "Catname": "X5000",
    "Catorder": "Military",
    "Industry": "Other",
    "Sector": "Other",
    "Sector Long": "Other"
  },
  {
    "Catcode": "W05",
    "Catname": "X7000",
    "Catorder": "Churches, clergy & religious organizations",
    "Industry": "Clergy & Religious Organizations",
    "Sector": "Other",
    "Sector Long": "Other"
  },
  {
    "Catcode": "W07",
    "Catname": "X8000",
    "Catorder": "International Organizations",
    "Industry": "Other",
    "Sector": "Other",
    "Sector Long": "Other"
  },
  {
    "Catcode": "W07",
    "Catname": "X9000",
    "Catorder": "Foreign Governments",
    "Industry": "Other",
    "Sector": "Other",
    "Sector Long": "Other"
  },
  {
    "Catcode": "Y00",
    "Catname": "Y0000",
    "Catorder": "Unknown",
    "Industry": "Unknown",
    "Sector": "Unknown",
    "Sector Long": "Unknown"
  },
  {
    "Catcode": "Y01",
    "Catname": "Y1000",
    "Catorder": "Homemakers, students & other non-income earners",
    "Industry": "Homemakers/Non-income earners",
    "Sector": "Unknown",
    "Sector Long": "Unknown"
  },
  {
    "Catcode": "Y02",
    "Catname": "Y2000",
    "Catorder": "No employer listed or discovered",
    "Industry": "No Employer Listed or Found",
    "Sector": "Unknown",
    "Sector Long": "Unknown"
  },
  {
    "Catcode": "Y03",
    "Catname": "Y3000",
    "Catorder": "Generic occupation - impossible to assign category",
    "Industry": "Generic Occupation/Category Unknown",
    "Sector": "Unknown",
    "Sector Long": "Unknown"
  },
  {
    "Catcode": "Y04",
    "Catname": "Y4000",
    "Catorder": "Employer listed but category unknown",
    "Industry": "Employer Listed/Category Unknown",
    "Sector": "Unknown",
    "Sector Long": "Unknown"
  },
  {
    "Catcode": "Q16",
    "Catname": "Z1000",
    "Catorder": "Candidate Committees",
    "Industry": "Candidate Committees",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q16",
    "Catname": "Z1100",
    "Catorder": "Republican Candidate Committees",
    "Industry": "Candidate Committees",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q16",
    "Catname": "Z1200",
    "Catorder": "Democratic Candidate Committees",
    "Industry": "Candidate Committees",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q16",
    "Catname": "Z1300",
    "Catorder": "Third-Party Candidate Committees",
    "Industry": "Candidate Committees",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Q16",
    "Catname": "Z1400",
    "Catorder": "Unknown-Party Candidate Committees",
    "Industry": "Candidate Committees",
    "Sector": "Ideology/Single-Issue",
    "Sector Long": "Ideological/Single-Issue"
  },
  {
    "Catcode": "Z04",
    "Catname": "Z4100",
    "Catorder": "Republican Joint Candidate Committee",
    "Industry": "Joint Candidate Cmte",
    "Sector": "Joint Candidate Cmtes",
    "Sector Long": "Joint Candidate Cmtes"
  },
  {
    "Catcode": "Z04",
    "Catname": "Z4200",
    "Catorder": "Democratic Joint Candidate Committee",
    "Industry": "Joint Candidate Cmte",
    "Sector": "Joint Candidate Cmtes",
    "Sector Long": "Joint Candidate Cmtes"
  },
  {
    "Catcode": "Z04",
    "Catname": "Z4300",
    "Catorder": "Third-Party Joint Candidate Committee",
    "Industry": "Joint Candidate Cmte",
    "Sector": "Joint Candidate Cmtes",
    "Sector Long": "Joint Candidate Cmtes"
  },
  {
    "Catcode": "Z04",
    "Catname": "Z4400",
    "Catorder": "Liberal Non-party Joint Fundraising Committee",
    "Industry": "Joint Candidate Cmte",
    "Sector": "Joint Candidate Cmtes",
    "Sector Long": "Joint Candidate Cmtes"
  },
  {
    "Catcode": "Z04",
    "Catname": "Z4500",
    "Catorder": "Conservative Non-party Joint Fundraising Committee",
    "Industry": "Joint Candidate Cmte",
    "Sector": "Joint Candidate Cmtes",
    "Sector Long": "Joint Candidate Cmtes"
  },
  {
    "Catcode": "Z02",
    "Catname": "Z5000",
    "Catorder": "Party Committees",
    "Industry": "Party Committees",
    "Sector": "Party Cmte",
    "Sector Long": "Party Cmtes"
  },
  {
    "Catcode": "Z02",
    "Catname": "Z5100",
    "Catorder": "Republican Party Committees",
    "Industry": "Party Committees",
    "Sector": "Party Cmte",
    "Sector Long": "Party Cmtes"
  },
  {
    "Catcode": "Z02",
    "Catname": "Z5200",
    "Catorder": "Democratic Party Committees",
    "Industry": "Party Committees",
    "Sector": "Party Cmte",
    "Sector Long": "Party Cmtes"
  },
  {
    "Catcode": "Z02",
    "Catname": "Z5300",
    "Catorder": "Third-Party Party Committees",
    "Industry": "Party Committees",
    "Sector": "Party Cmte",
    "Sector Long": "Party Cmtes"
  },
  {
    "Catcode": "Z07",
    "Catname": "Z9000",
    "Catorder": "Candidate contribution to his/her own campaign",
    "Industry": "Candidate Self-finance",
    "Sector": "Candidate",
    "Sector Long": "Candidate"
  },
  {
    "Catcode": "Z08",
    "Catname": "Z9100",
    "Catorder": "Transfer between national party committees",
    "Industry": "Party Committee Transfer",
    "Sector": "Non-contribution",
    "Sector Long": "Non-contribution"
  },
  {
    "Catcode": "Z09",
    "Catname": "Z9500",
    "Catorder": "Transfer from intermediary (type 24I or 24T)",
    "Industry": "Non-contribution",
    "Sector": "Non-contribution",
    "Sector Long": "Non-contribution"
  },
  {
    "Catcode": "Z09",
    "Catname": "Z9600",
    "Catorder": "Non-Contribution, Miscellaneous",
    "Industry": "Non-contribution",
    "Sector": "Non-contribution",
    "Sector Long": "Non-contribution"
  },
  {
    "Catcode": "Z09",
    "Catname": "Z9700",
    "Catorder": "Unitemized (small) contributions",
    "Industry": "Non-contribution",
    "Sector": "Non-contribution",
    "Sector Long": "Non-contribution"
  },
  {
    "Catcode": "Z09",
    "Catname": "Z9800",
    "Catorder": "Public Funding",
    "Industry": "Non-contribution",
    "Sector": "Non-contribution",
    "Sector Long": "Non-contribution"
  },
  {
    "Catcode": "Z09",
    "Catname": "Z9999",
    "Catorder": "Internal Transfer and other non-contributions",
    "Industry": "Non-contribution",
    "Sector": "Non-contribution",
    "Sector Long": "Non-contribution"
  }
  ];







  

 function  divmagic(d) {
   
  var display = ["#inf", "#sen1", "#officials", "#content", "#about", "#lobby", "#contact", "#carousel", "#caroucontent", 
              "#cards", "#plan", "#sen1", "#sen1data", "#sen2", "#sen2data", "#rep", "#repdata", "#getlobbydata", "#inf", "#repcontent", "#NavbarLogo"];
  
  var empty = ["#sen1", "#sen1committeedatabutton","#sen1subcommitteedatabutton", "#sen2", "#sen2committeedatabutton",
              "#sen2subcommitteedatabutton", "#rep", "#repcommitteedatabutton", "#repsubcommitteedatabutton", "#D", "#R" ]
        
        
          for (k=0; k<=display.length; k++){
          if (k!==d){
            $(d).css('display', 'none');
          } else {
            $(d).css('display', 'block').fadeIn(2000);
          }
        //document.getElementById(d).css('display', 'block');
     };
     //lets save the empty relationship for ya mom you slut another day 
     
     };


// LINK TOILETS 

// remember bro this one got repurposed. this isn't one of the banner links anymore
$(document).ready(function(){
    $('#getlobbydata').click(function(){
      $('#contact').css('display', 'none')
      $('#officials').css('display', 'none')
      $('#about').css('display', 'none')
      $('#home').css('display', 'none')
      $('#lobby').css('display', 'none')
      $('#carousel').fadeOut(2000)
      $('#caroucontent').fadeOut(2000)
      $('#inf').empty()
      $('#inf').css('display', 'none')
    })
  });
  
  // this is where you began the jQuery css display automation function from 
  //Home
  $(document).ready(function(){
    $('#NavbarLogo').click(function(){
      divmagic("#NavbarLogo");
      // $('#home').fadeIn(2000)
      // $('#officials').fadeIn(2000).css('display', 'block')
      // $('#firstcontent').fadeIn(2000)
      // $('#content').css('display', 'none')
      // $('#about').css('display', 'none')
      // $('#lobby').css('display', 'none')
      // $('#contact').css('display', 'none')
      // $('#carousel').css('display', 'none')
      // $('#caroucontent').css('display', 'none')
      // $('#cards').css('display', 'none')
      // $('#plan').css('display', 'none')
      // $('#sen1').css('display', 'none')
      // $('#sen1').empty()
      // $('#sen1data').css('display', 'none')
      // $('#sen1committeedatabutton').empty()
      // $('#sen1subcommitteedatabutton').empty()
      // $('#sen2').css('display', 'none')
      // $('#sen2').empty()
      // $('#sen2data').css('display', 'none')
      // $('#sen2committeedatabutton').empty()
      // $('#sen2subcommitteedatabutton').empty()
      // $('#rep').css('display', 'none')
      // $('#rep').empty()
      // $('#repcommitteedatabutton').empty()
      // $('#repsubcommitteedatabutton').empty()
      // $('#repdata').css('display', 'none')
      // $('#getlobbydata').css('display', 'none')
      // $('#D').empty()
      // $('#D').css('display', 'none')
      // $('#R').empty()
      // $('#R').css('display', 'none')
      // $('#inf').empty()
      // $('#inf').css('display', 'none')
      // $('#sen1table').empty()
      // $('#sen2table').empty()
      // $('#reptable').empty()
      // $('#repcontent').css('display', 'none')
      // $('#load').css('display', 'none')
    })
  });
  //Sources
  $(document).ready(function(){
    $('#About').click(function(){
      $('#about').fadeIn(3000).css('display', 'block')
      $('#officials').css('display', 'none')
      $('#content').css('display', 'none')
      $('#contact').css('display', 'none')
      $('#home').css('display', 'none')
      $('#lobby').css('display', 'none')
      $('#getlobbydata').css('display', 'none')
      $('#carousel').css('display', 'none')
      $('#caroucontent').css('display', 'none')
      $('#cards').css('display', 'none')
      $('#plan').css('display', 'none')
      $('#sen1').css('display', 'none')
      $('#sen1').empty()
      $('#sen1data').css('display', 'none')
      $('#sen1committeedatabutton').empty()
      $('#sen1subcommitteedatabutton').empty()
      $('#sen2').css('display', 'none')
      $('#sen2').empty()
      $('#sen2data').css('display', 'none')
      $('#sen2committeedatabutton').empty()
      $('#sen2subcommitteedatabutton').empty()
      $('#rep').css('display', 'none')
      $('#rep').empty()
      $('#repcommitteedatabutton').empty()
      $('#repsubcommitteedatabutton').empty()
      $('#repdata').css('display', 'none')
      $('#D').empty()
      $('#R').empty()
      $('#inf').empty()
      $('#inf').css('display', 'none')
      $('#sen1table').empty()
      $('#sen2table').empty()
      $('#reptable').empty()
      $('#R').css('display', 'none')
      $('#D').css('display', 'none')
      $('#repcontent').css('display', 'none')
    })
  });
  //Contact
  $(document).ready(function(){
    $('#Contact').click(function(){
      $('#contact').fadeIn(3000).css('display', 'block')
      $('#officials').css('display', 'none')
      $('#content').css('display', 'none')
      $('#about').css('display', 'none')
      $('#home').css('display', 'none')
      $('#lobby').css('display', 'none')
      $('#getlobbydata').css('display', 'none')
      $('#carousel').css('display', 'none')
      $('#caroucontent').css('display', 'none')
      $('#cards').css('display', 'none')
      $('#plan').css('display', 'none')
      $('#sen1').css('display', 'none')
      $('#sen1').empty()
      $('#sen1data').css('display', 'none')
      $('#sen1committeedatabutton').empty()
      $('#sen1subcommitteedatabutton').empty()
      $('#sen2').css('display', 'none')
      $('#sen2').empty()
      $('#sen2data').css('display', 'none')
      $('#sen2committeedatabutton').empty()
      $('#sen2subcommitteedatabutton').empty()
      $('#rep').css('display', 'none')
      $('#rep').empty()
      $('#repcommitteedatabutton').empty()
      $('#repsubcommitteedatabutton').empty()
      $('#repdata').css('display', 'none')
      $('#D').empty()
      $('#R').empty()
      $('#inf').empty()
      $('#inf').css('display', 'none')
      $('#sen1table').empty()
      $('#sen2table').empty()
      $('#reptable').empty()
      $('#R').css('display', 'none')
      $('#D').css('display', 'none')
      $('#repcontent').css('display', 'none')
      $.ajax({
        url: "https://www.googleapis.com/geolocation/v1/geolocate?key=AIzaSyCz9er75qmCoQA7RXxC15skKRtgP-eWuoA",
        type: "POST",
        dataType: 'json'
    }).done(function(data){
      console.log(data);
    })
    })
  });

  //officials
  $(document).ready(function() {
    $("#officials").click(function(){
      $("#lobby").fadeIn(4000)
      $('#cards').delay(1000).fadeIn(4000)
      $('#content').fadeIn(1000)
      $('#committeeshit').css('display', 'none'),
      $('#officials').css('display', 'none')
      $('#home').css('display', 'none')
      $('#about').css('display', 'none')
      $('#contact').css('display', 'none')
});
});
  // MAPs 

  // MAPS
  

  
  //Finding Congressman ID Function
  function ff(x) {
  var IDs= 
  {"James Abdnor":"A000009","Neil Abercrombie":"A000014","James Abourezk":"A000017","Ralph Lee Abraham":"A000374","Spencer Abraham":"A000355","Bella S. Abzug":"A000018","Anibal Acevedo-Vila":"A000359","Gary L. Ackerman":"A000022","Alma S. Adams":"A000370","Brock Adams":"A000031","Sandy Adams":"A000366","Joseph P. Addabbo":"A000052","Robert B. Aderholt":"A000055","John H. Adler":"A000364","Pete Aguilar":"A000371","George Aiken":"A000062","Daniel K. Akaka":"A000069","W. Todd Akin":"A000358","Carl Albert":"A000073","Donald J. Albosta":"A000076","Bill Alexander":"A000103","Lamar Alexander":"A000360","Rodney Alexander":"A000361","Wayne Allard":"A000109","Clifford R. Allen":"A000118","George Allen":"A000121","James B. Allen":"A000127","Maryon Allen":"A000139","Rick W. Allen":"A000372","Thomas H. Allen":"A000357","Colin Z. Allred":"A000376","Jason Altmire":"A000362","Justin Amash":"A000367","Jerome A. Ambro":"A000170","Joseph S. Ammerman":"A000177","Mark E. Amodei":"A000369","Glenn M. Anderson":"A000189","John B. Anderson":"A000195","Wendell R. Anderson":"A000202","Ike Andrews":"A000207","Mark Andrews":"A000208","Michael Andrews":"A000209","Robert E. Andrews":"A000210","Thomas H. Andrews":"A000211","Frank Annunzio":"A000212","Jr. Anthony, Beryl":"A000213","Douglas Applegate":"A000214","Bill Archer":"A000215","Michael A. Arcuri":"A000363","Leslie C. Arends":"A000216","Richard K. Armey":"A000217","Kelly Armstrong":"A000377","William L. Armstrong":"A000219","Jodey C. Arrington":"A000375","Jean Ashbrook":"A000220","John M. Ashbrook":"A000221","John Ashcroft":"A000356","Brad Ashford":"A000373","Thomas L. Ashley":"A000222","Les Aspin":"A000224","Chester G. Atkins":"A000226","Eugene V. Atkinson":"A000329","Les AuCoin":"A000337","Steve Austria":"A000365","Cynthia Axne":"A000378","Kelly Ayotte":"A000368","Brian Babin":"B001291","Joe Baca":"B001234","Jim Bacchus":"B000008","Michele Bachmann":"B001256","Spencer Bachus":"B000013","Don Bacon":"B001298","Robert E. Badham":"B000024","Herman Badillo":"B000025","Scotty Baesler":"B000028","L. A. (Skip) Bafalis":"B000029","Donald A. Bailey":"B000037","Wendell Bailey":"B000047","Brian Baird":"B001229","James R. Baird":"B001307","Bill Baker":"B000078","Jr. Baker, Howard H.":"B000063","LaMar Baker":"B000069","Richard H. Baker":"B000072","John Elias Baldacci":"B000081","Troy Balderson":"B001306","Alvin J. Baldus":"B000083","Tammy Baldwin":"B001230","Jr. Ballance, Frank W.":"B001238","Cass Ballenger":"B000104","Jim Banks":"B001299","Ron Barber":"B001279","Peter Barca":"B001226","James A. Barcia":"B000134","Dean M. Barkley":"B001237","Lou Barletta":"B001269","Tom Barlow":"B000151","Jr. Barnard, Doug":"B000153","Michael D. Barnes":"B000160","Andy Barr":"B001282","Bob Barr":"B000169","Nanette Diaz Barragan":"B001300","John Barrasso":"B001261","Bill Barrett":"B000179","J. Gresham Barrett":"B001239","Thomas M. Barrett":"B000177","William A. Barrett":"B000178","John Barrow":"B001252","Dewey F. Bartlett":"B000200","Roscoe G. Bartlett":"B000208","Steve Bartlett":"B000204","Joe Barton":"B000213","Charles F. Bass":"B000220","Karen Bass":"B001270","Herbert H. Bateman":"B000229","Jim Bates":"B000236","Max Baucus":"B000243","Robert E. Bauman":"B000244","Birch Bayh":"B000254","Evan Bayh":"B001233","Jr. Beall, J. Glenn":"B000272","Melissa L. Bean":"B001253","Edward Beard":"B000279","Robin Beard":"B000280","Joyce Beatty":"B001281","Bob Beauprez":"B001240","Xavier Becerra":"B000287","Berkley W. Bedell":"B000298","Mark Begich":"B001265","Anthony C. Beilenson":"B000318","Alphonzo Bell":"B000330","Chris Bell":"B001241","Henry L. Bellmon":"B000351","Cleve Benedict":"B000358","Dan Benishek":"B001271","Jamie Benitez":"B000362","Jr. Benjamin, Adam":"B000363","Michael F. Bennet":"B001267","Charles E. Bennett":"B000371","Robert F. Bennett":"B000382","Wallace F. Bennett":"B000384","Kerry L. Bentivolio":"B001280","Helen Delich Bentley":"B000392","Ken Bentsen":"B000400","Lloyd M. Bentsen":"B000401","Ami Bera":"B001287","Doug Bereuter":"B000403","Rick Berg":"B001272","Bob Bergland":"B000408","Jack Bergman":"B001301","Shelley Berkley":"B001231","Howard L. Berman":"B000410","Marion Berry":"B000420","Ed Bethune":"B000422","Tom Bevill":"B000431","Jr. Beyer, Donald S.":"B001292","Mario Biaggi":"B000432","Alan Bible":"B000436","Jr. Biden, Joseph R.":"B000444","Jr. Biester, Edward G.":"B000451","Judy Biggert":"B001232","Andy Biggs":"B001302","Brian P. Bilbray":"B000461","James H. Bilbray":"B000462","Gus M. Bilirakis":"B001257","Michael Bilirakis":"B000463","Jeff Bingaman":"B000468","Jonathan B. Bingham":"B000472","Mike Bishop":"B001293","Rob Bishop":"B001250","Jr. Bishop, Sanford D.":"B000490","Timothy H. Bishop":"B001242","Diane Black":"B001273","Ben Blackburn":"B000506","Marsha Blackburn":"B001243","Lucien E. Blackwell":"B000517","Rod R. Blagojevich":"B000518","James Blanchard":"B000538","John Anton Blatnik":"B000550","Ben G. Blaz":"B000551","Tom Bliley":"B000556","Michael Blouin":"B000567","Rod Blum":"B001294","Earl Blumenauer":"B000574","Richard Blumenthal":"B001277","Lisa Blunt Rochester":"B001303","Roy Blunt":"B000575","Peter I. Blute":"B000576","John A. Boccieri":"B001263","Sherwood Boehlert":"B000586","John A. Boehner":"B000589","Corinne C. (Lindy) Boggs":"B000592","Edward P. Boland":"B000600","Richard Bolling":"B000605","Suzanne Bonamici":"B001278","Christopher S. Bond":"B000611","William H. Boner":"B000615","Henry Bonilla":"B000617","David E. Bonior":"B000619","Don Bonker":"B000620","Jo Bonner":"B001244","Mary Bono Mack":"B001228","Sonny Bono":"B000622","Cory A. Booker":"B001288","John Boozman":"B001236","Madeleine Z. Bordallo":"B001245","Dan Boren":"B001254","David L. Boren":"B000639","Robert A. Borski":"B000644","Rudy Boschwitz":"B000647","Douglas H. Bosco":"B000648","Mike Bost":"B001295","Leonard L. Boswell":"B000652","Rick Boucher":"B000657","Beau Boulter":"B000666","Jr. Boustany, Charles W.":"B001255","David R. Bowen":"B000682","Barbara Boxer":"B000711","Allen Boyd":"B000716","Nancy E. Boyda":"B001258","Brendan F. Boyle":"B001296","John Brademas":"B000736","Bill Bradley":"B001225","Jeb Bradley":"B001246","Kevin Brady":"B000755","Nicholas Brady":"B000756","Robert A. Brady":"B001227","Bruce L. Braley":"B001259","Frank Brasco":"B000771","Dave Brat":"B001290","Mike Braun":"B001310","William G. Bray":"B000778","John B. Breaux":"B000780","John B. Breckinridge":"B000788","Joseph E. Brennan":"B000798","Bill K. Brewster":"B000817","Jim Bridenstine":"B001283","Bobby Bright":"B001264","Anthony Brindisi":"B001308","Jack Brinkley":"B000839","Robin Britt":"B000845","Bill Brock":"B000851","William Brodhead":"B000862","Edward W. Brooke":"B000871","Jack B. Brooks":"B000880","Mo Brooks":"B001274","Susan W. Brooks":"B001284","William S. Broomfield":"B000890","Donald G. Brotzman":"B000893","Paul C. Broun":"B001262","Glen Browder":"B000897","Anthony G. Brown":"B001304","Jr. Brown, Clarence":"B000910","Corrine Brown":"B000911","Garry E. Brown":"B000917","Jr. Brown, George E.":"B000918","Hank Brown":"B000919","Jr. Brown, Henry E.":"B001235","Scott P. Brown":"B001268","Sherrod Brown":"B000944","Ginny Brown-Waite":"B001247","Sam Brownback":"B000953","Julia Brownley":"B001285","James T. Broyhill":"B000966","Joel T. Broyhill":"B000967","Terry L. Bruce":"B000971","Richard H. Bryan":"B000993","Ed Bryant":"B000996","John W. Bryant":"B000997","John Buchanan":"B001008","Vern Buchanan":"B001260","Ken Buck":"B001297","James L. Buckley":"B001026","Larry Bucshon":"B001275","Ted Budd":"B001305","Jack Buechner":"B001036","Ann Marie Buerkle":"B001276","Dale Bumpers":"B001057","Jim Bunn":"B001063","Jim Bunning":"B001066","Tim Burchett":"B001309","Joycelyn Burdick":"B001076","Quentin N. Burdick":"B001077","Clair W. Burgener":"B001080","Michael C. Burgess":"B001248","J. Herbert Burke":"B001091","James A. Burke":"B001092","Yvonne B. Burke":"B001102","Omar Burleson":"B001111","Bill D. Burlison":"B001113","Conrad R. Burns":"B001126","Max Burns":"B001249","Richard Burr":"B001135","Roland Burris":"B001266","Dan Burton":"B001149","John Burton":"B001153","Phillip Burton":"B001156","Sala Burton":"B001158","Albert G. Bustamante":"B001172","Cheri Bustos":"B001286","M. Caldwell Butler":"B001182","G. K. Butterfield":"B001251","Steve Buyer":"B001203","Jr. Byrd, Harry F.":"B001209","Robert C. Byrd":"B001210","Bradley Byrne":"B001289","Leslie L. Byrne":"B001213","Beverly B. Byron":"B001220","Goodloe E. Byron":"B001221","Sonny Callahan":"C000052","Ken Calvert":"C000059","Dave Camp":"C000071","John N. Happy Camp":"C000073","Ben Nighthorse Campbell":"C000077","Jr. Campbell, Carroll A.":"C000079","John Campbell":"C001064","Tom Campbell":"C000100","Charles T. Canady":"C000107","Chris Cannon":"C000116","Howard W. Cannon":"C000120","Francisco \"Quico\" Canseco":"C001082","Eric Cantor":"C001046","Maria Cantwell":"C000127","Anh \"Joseph\" Cao":"C001079","Shelley Moore Capito":"C001047","Lois Capps":"C001036","Walter Capps":"C000134","Michael E. Capuano":"C001037","Bruce F. Caputo":"C000137","Salud O. Carbajal":"C001112","Tony Cardenas":"C001097","Benjamin L. Cardin":"C000141","Dennis A. Cardoza":"C001050","Hugh L. Carey":"C000143","Gregory W. Carman":"C000158","Jean Carnahan":"C001043","Russ Carnahan":"C001060","Charles J. Carney":"C000164","Christopher P. Carney":"C001065","Jr. Carney, John C.":"C001083","William Carney":"C000165","Thomas R. Carper":"C000174","Bob Carr":"C000178","Andre Carson":"C001072","Brad Carson":"C001044","Julia Carson":"C000191","Earl L. \"Buddy\" Carter":"C001103","John R. Carter":"C001051","Tim Lee Carter":"C000201","Matt Cartwright":"C001090","Clifford P. Case":"C000220","Ed Case":"C001055","Bob Casey":"C000228","Jr. Casey, Robert P.":"C001070","Bill Cassidy":"C001075","Sean Casten":"C001117","Michael N. Castle":"C000243","Kathy Castor":"C001066","Joaquin Castro":"C001091","John J. Cavanaugh":"C000261","Jr. Cazayoux, Donald J.":"C001073","Elford A. Cederberg":"C000263","Steve Chabot":"C000266","John H. Chafee":"C000269","Lincoln Chafee":"C001040","Jason Chaffetz":"C001076","Charles E. Chamberlain":"C000275","Saxby Chambliss":"C000286","Ben Chandler":"C001058","Rod D. Chandler":"C000293","Jim Chapman":"C000312","Jr. Chappell, Bill":"C000321","Eugene A. Chappie":"C000322","Dick Cheney":"C000344","Liz Cheney":"C001109","Helen Chenoweth-Hage":"C000345","Jeff Chiesa":"C001100","Travis Childers":"C001074","Lawton Chiles":"C000356","Shirley Chisholm":"C000371","Chris Chocola":"C001052","Donna M. Christensen":"C000380","Jon Christensen":"C000377","Dick Chrysler":"C000385","Judy Chu":"C001080","Frank Church":"C000388","David N. Cicilline":"C001084","Jr. Cisneros, Gilbert Ray":"C001123","Donald D. Clancy":"C000409","Dick Clark":"C000448","Frank M. Clark":"C000431","Katherine M. Clark":"C001101","Hansen Clarke":"C001085","James McC. Clarke":"C000462","Yvette D. Clarke":"C001067","Don H. Clausen":"C000475","Curt Clawson":"C001102","Del Clawson":"C000476","William (Bill) Clay":"C000488","Wm. Lacy Clay":"C001049","Eva M. Clayton":"C000494","Emanuel Cleaver":"C001061","Max Cleland":"C001034","Bob Clement":"C000503","James C. Cleveland":"C000512","Ben Cline":"C001118","Jr. Clinger, William F.":"C000523","Hillary Rodham Clinton":"C001041","Michael Cloud":"C001115","James E. Clyburn":"C000537","Daniel Coats":"C000542","William W. Cobey":"C000555","Howard Coble":"C000556","Tom Coburn":"C000560","Thad Cochran":"C000567","Anthony Lee Coelho":"C000581","Mike Coffman":"C001077","Steve Cohen":"C001068","William S. Cohen":"C000598","Tom Cole":"C001053","E. Thomas Coleman":"C000618","Norm Coleman":"C001057","Ronald D. Coleman":"C000621","Harold R. Collier":"C000629","Barbara-Rose Collins":"C000633","Cardiss Collins":"C000634","Chris Collins":"C001092","Doug Collins":"C001093","James M. Collins":"C000638","Mac Collins":"C000640","Susan M. Collins":"C001035","Antonio Colordao":"C000646","Larry Combest":"C000653","James Comer":"C001108","Barbara Comstock":"C001105","Jr. Conable, Barber B.":"C000666","K. Michael Conaway":"C001062","Gary A. Condit":"C000670","John B. Conlan":"C000682","Gerald E. Connolly":"C001078","Kent Conrad":"C000705","Silvio O. Conte":"C000709","Jr. Conyers, John":"C000714","Marlow W. Cook":"C000721","Merrill Cook":"C000722","Paul Cook":"C001094","John Cooksey":"C000735","Wes Cooley":"C000737","Christopher A. Coons":"C001088","Jim Cooper":"C000754","Sam Coppersmith":"C000767","Tom Corcoran":"C000773","Bob Corker":"C001071","James C. Corman":"C000780","Robert J. Cornell":"C000782","David L. Cornwell":"C000787","John Cornyn":"C001056","Baltasar Corrada":"C000788","J. Luis Correa":"C001110","Catherine Cortez Masto":"C001113","Jon S. Corzine":"C001042","Jim Costa":"C001059","Jerry F. Costello":"C000794","Ryan A. Costello":"C001106","William R. Cotter":"C000799","Norris Cotton":"C000802","Tom Cotton":"C001095","Lawrence Coughlin":"C000807","Jim Courter":"C000809","Joe Courtney":"C001069","Paul Coverdell":"C000813","William M. Cowan":"C001099","Christopher Cox":"C000830","Jr. Cox, John W.":"C000836","TJ Cox":"C001124","James K. Coyne":"C000845","William J. Coyne":"C000846","Angie Craig":"C001119","Larry E. Craig":"C000858","Kevin Cramer":"C001096","Jr. Cramer, Robert E. (Bud)":"C000868","Daniel B. Crane":"C000871","Philip M. Crane":"C000873","Alan Cranston":"C000877","Mike Crapo":"C000880","Chip Cravaack":"C001086","Eric A. \"Rick\" Crawford":"C001087","Frank A. Cremeans":"C000903","Ander Crenshaw":"C001045","Dan Crenshaw":"C001120","Charlie Crist":"C001111","Mark S. Critz":"C001081","Jr. Crockett, George W.":"C000919","Paul W. Cronin":"C000925","Jason Crow":"C001121","Joseph Crowley":"C001038","Ted Cruz":"C001098","Barbara Cubin":"C000962","Henry Cuellar":"C001063","John Abney Culberson":"C001048","John C. Culver":"C000979","Elijah E. Cummings":"C000984","Joe Cunningham":"C001122","John E. (Jack) Cunningham":"C000992","Randy (Duke) Cunningham":"C000994","Carlos Curbelo":"C001107","David Alan Curson":"C001089","Carl T. Curtis":"C001006","John R. Curtis":"C001114","Alfonse D'Amato":"D000018","Norman E. D'Amours":"D000017","Kathleen A. Dahlkemper":"D000608","Steve Daines":"D000618","John C. Danforth":"D000030","Jr. Daniel, Robert W.":"D000037","W. C. (Dan) Daniel":"D000038","Dominick V. Daniels":"D000041","George E. Danielson":"D000043","William E. Dannemeyer":"D000044","Pat Danner":"D000046","George (Buddy) Darden":"D000051","Thomas A. Daschle":"D000064","Hal Daub":"D000065","Sharice Davids":"D000629","Warren Davidson":"D000626","Artur Davis":"D000602","Danny K. Davis":"D000096","David Davis":"D000606","Geoff Davis":"D000603","Glenn R. Davis":"D000102","Jack Davis":"D000106","Jim Davis":"D000114","Jo Ann Davis":"D000597","John W. Davis":"D000122","Lincoln Davis":"D000599","Mendel J. Davis":"D000125","Robert W. Davis":"D000131","Rodney Davis":"D000619","Susan A. Davis":"D000598","Tom Davis":"D000136","Mark Dayton":"D000596","E. de la Garza":"D000203","Ron de Lugo":"D000209","Nathan Deal":"D000168","Madeleine Dean":"D000631","Joel Deckard":"D000183","Dennis DeConcini":"D000185","Peter A. DeFazio":"D000191","Diana DeGette":"D000197","Bill Delahunt":"D000210","James J. Delaney":"D000211","John K. Delaney":"D000620","Rosa L. DeLauro":"D000216","Tom DeLay":"D000217","Suzan K. DelBene":"D000617","Antonio Delgado":"D000630","John Dellenback":"D000220","Ronald V. Dellums":"D000222","Val Butler Demings":"D000627","Jim DeMint":"D000595","Lawrence J. DeNardis":"D000231","Jeff Denham":"D000612","Frank E. Denholm":"D000234","David W. Dennis":"D000241","Charles W. Dent":"D000604","John H. Dent":"D000255","Jeremiah Denton":"D000259","Butler C. Derrick":"D000267","Edward J. Derwinski":"D000269","Ron DeSantis":"D000621","Mark DeSaulnier":"D000623","Scott DesJarlais":"D000616","Theodore E. Deutch":"D000610","Peter Deutsch":"D000275","Samuel L. Devine":"D000279","Mike DeWine":"D000294","Lincoln Diaz-Balart":"D000299","Mario Diaz-Balart":"D000600","Jay Dickey":"D000312","William L. Dickinson":"D000326","Norman D. Dicks":"D000327","Jr. Diggs, Charles C.":"D000344","Debbie Dingell":"D000624","John D. Dingell":"D000355","Joseph J. DioGuardi":"D000359","Alan J. Dixon":"D000366","Julian C. Dixon":"D000373","Charles K. Djou":"D000611","Christopher J. Dodd":"D000388","Lloyd Doggett":"D000399","Robert J. Dold":"D000613","Elizabeth Dole":"D000601","Robert J. Dole":"D000401","Pete V. Domenici":"D000407","Peter H. Dominick":"D000409","Brian J. Donnelly":"D000416","Joe Donnelly":"D000607","Harold D. Donohue":"D000419","Jr. Donovan, Daniel M.":"D000625","Calvin M. Dooley":"D000424","John T. Doolittle":"D000429","Byron L. Dorgan":"D000432","William Jennings Bryan Dorn":"D000434","Robert K. Dornan":"D000435","Charles F. Dougherty":"D000446","Chuck Douglas":"D000451","Wayne Dowdy":"D000466","Thomas J. Downey":"D000471","Thomas N. Downing":"D000474","Michael F. Doyle":"D000482","Thelma D. Drake":"D000605","David Dreier":"D000492","Steve Driehaus":"D000609","Robert F. Drinan":"D000499","IV du Pont, Pierre S.":"D000558","Tammy Duckworth":"D000622","Sean P. Duffy":"D000614","Thaddeus J. Dulski":"D000523","Jeff Duncan":"D000615","John J. Duncan":"D000534","Jr. Duncan, John J.":"D000533","Robert B. Duncan":"D000537","Jennifer Dunn":"D000549","Jim Dunn":"D000548","Neal P. Dunn":"D000628","Richard J. Durbin":"D000563","Dave Durenberger":"D000566","John A. Durkin":"D000574","Bernard J. Dwyer":"D000586","Mervyn M. Dymally":"D000592","Roy Dyson":"D000593","Thomas F. Eagleton":"E000004","Joseph D. Early":"E000013","John P. East":"E000017","James O. Eastland":"E000018","Dennis E. Eckart":"E000031","Fred J. Eckert":"E000033","Bob Eckhardt":"E000035","Robert W. Edgar":"E000043","Chet Edwards":"E000063","Don Edwards":"E000064","Donna F. Edwards":"E000290","Jack Edwards":"E000084","John Edwards":"E000286","Mickey Edwards":"E000077","Vernon J. Ehlers":"E000092","Jr. Ehrlich, Robert L.":"E000093","Joshua Eilberg":"E000096","Keith Ellison":"E000288","Renee L. Ellmers":"E000291","Brad Ellsworth":"E000289","Rahm Emanuel":"E000287","Bill Emerson":"E000174","Jo Ann Emerson":"E000172","David F. Emery":"E000175","Tom Emmer":"E000294","Eliot L. Engel":"E000179","Glenn English":"E000184","Karan English":"E000186","Phil English":"E000187","John Ensign":"E000194","William L. Enyart":"E000292","Michael B. Enzi":"E000285","Arlen Erdahl":"E000198","Ben Erdreich":"E000201","John N. Erlenborn":"E000204","Joni Ernst":"E000295","Allen E. Ertel":"E000208","Jr. Ervin, Sam J.":"E000211","Marvin L. Esch":"E000213","Veronica Escobar":"E000299","Edwin D. Eshleman":"E000214","Anna G. Eshoo":"E000215","Adriano Espaillat":"E000297","Mike Espy":"E000218","Ron Estes":"E000298","Elizabeth H. Esty":"E000293","Bob Etheridge":"E000226","Billy Lee Evans":"E000233","Cooper Evans":"E000259","Daniel J. Evans":"E000236","David W. Evans":"E000239","Dwight Evans":"E000296","Frank E. Evans":"E000240","Lane Evans":"E000250","Melvin H. Evans":"E000254","Jr. Evans, Thomas B.":"E000258","Terry Everett":"E000268","Joe L. Evins":"E000273","Thomas W. Ewing":"E000282","J. James Exon":"E000284","Lauch Faircloth":"F000437","Eni F. H. Faleomavaega":"F000010","Mary Fallin":"F000453","Paul J. Fannin":"F000013","Blake Farenthold":"F000460","Sam Farr":"F000030","John G. Fary":"F000040","Dante B. Fascell":"F000041","John J. Faso":"F000464","Chaka Fattah":"F000043","Walter E. Fauntroy":"F000046","Harris W. Fawell":"F000049","Vic Fazio":"F000053","Tom Feeney":"F000447","Edward F. Feighan":"F000059","Russell D. Feingold":"F000061","Dianne Feinstein":"F000062","Millicent H. Fenwick":"F000078","IV Ferguson, A. Drew":"F000465","Mike Ferguson":"F000443","Geraldine A. Ferraro":"F000088","Bobbi Fiedler":"F000102","Cleo Fields":"F000110","Jack Fields":"F000111","Bob Filner":"F000116","Stephen Lee Fincher":"F000458","Paul Findley":"F000123","Eric D. Fingerhut":"F000128","Abby Finkenauer":"F000467","Deb Fischer":"F000463","Jr. Fish, Hamilton":"F000141","Joesph L. Fisher":"F000151","O. C. Fisher":"F000152","Floyd J. Fithian":"F000161","Peter Fitzgerald":"F000442","Brian K. Fitzpatrick":"F000466","Michael G. Fitzpatrick":"F000451","Floyd H. Flake":"F000184","Jeff Flake":"F000444","Michael Patrick Flanagan":"F000187","Charles J. \"Chuck\" Fleischmann":"F000459","John Fleming":"F000456","Ernie Fletcher":"F000441","Lizzie Fletcher":"F000468","Ronnie G. Flippo":"F000208","Daniel J. Flood":"F000209","Bill Flores":"F000461","James J. Florio":"F000215","Walter Flowers":"F000218","Jr. Flynt, John J.":"F000229","Thomas M. Foglietta":"F000235","Mark Foley":"F000238","Thomas S. Foley":"F000239","Hiram L. Fong":"F000245","J. Randy Forbes":"F000445","Michael P. Forbes":"F000257","Jr. Ford, Gerald R.":"F000260","Harold E. Ford":"F000261","Jr. Ford, Harold E.":"F000262","Wendell H. Ford":"F000268","William D. Ford":"F000270","Edwin B. Forsythe":"F000286","Jeff Fortenberry":"F000449","Luis G. Fortuno":"F000452","Vito Fossella":"F000440","Bill Foster":"F000454","L. H. Fountain":"F000319","Tillie Fowler":"F000328","Jr. Fowler, Wyche":"F000329","Jon D. Fox":"F000332","Virginia Foxx":"F000450","Sheila Frahm":"F000438","Barney Frank":"F000339","Lois Frankel":"F000462","Al Franken":"F000457","William W. Franklin":"F000347","Bob Franks":"F000349","Gary A. Franks":"F000348","Trent Franks":"F000448","Donald M. Fraser":"F000350","Victor O. Frazer":"F000351","Peter H. B. Frelinghuysen":"F000371","Rodney P. Frelinghuysen":"F000372","Bill Frenzel":"F000380","Jr. Frey, Lou":"F000381","Dan Frisa":"F000387","William H. Frist":"F000439","Harold V. Froehlich":"F000388","Martin Frost":"F000392","Marcia L. Fudge":"F000455","James Fulbright":"F000401","Russ Fulcher":"F000469","Richard Fulton":"F000424","David Funderburk":"F000426","Don Fuqua":"F000430","Elizabeth Furse":"F000434","Jamie B. Fuster":"F000435","Tulsi Gabbard":"G000571","Matt Gaetz":"G000578","Mike Gallagher":"G000579","Elton Gallegly":"G000021","Pete P. Gallego":"G000572","Ruben Gallego":"G000574","Dean A. Gallo":"G000025","Robert A. Gammage":"G000036","Greg Ganske":"G000041","John Garamendi":"G000559","Jesus G. \"Chuy\" Garcia":"G000586","Joe Garcia":"G000573","Robert Garcia":"G000047","Sylvia R. Garcia":"G000587","Cory Gardner":"G000562","E. J. (Jake) Garn":"G000072","Scott Garrett":"G000548","Jr. Garrett, Thomas A.":"G000580","Joseph M. Gaydos":"G000105","Sam Gejdenson":"G000120","George W. Gekas":"G000121","Richard A. Gephardt":"G000132","Pete Geren":"G000134","Jim Gerlach":"G000549","Tom S. Gettys":"G000144","Robert N. Giaimo":"G000151","Greg Gianforte":"G000584","Jim Gibbons":"G000152","Sam Gibbons":"G000153","Bob Gibbs":"G000563","Christopher P. Gibson":"G000564","Gabrielle Giffords":"G000554","Wayne T. Gilchrest":"G000180","Kirsten E. Gillibrand":"G000555","Paul E. Gillmor":"G000210","Benjamin A. Gilman":"G000212","Phil Gingrey":"G000550","Newt Gingrich":"G000225","Bo Ginn":"G000226","Jr. Glenn, John H.":"G000236","Dan Glickman":"G000240","Louie Gohmert":"G000552","Jared F. Golden":"G000592","Barry Goldwater":"G000267","Jr. Goldwater, Barry":"G000268","Jimmy Gomez":"G000585","Anthony Gonzalez":"G000588","Charles A. Gonzalez":"G000544","Henry B. Gonzalez":"G000272","Vicente Gonzalez":"G000581","Jenniffer Gonzalez-Colon":"G000582","Jr. Goode, Virgil H.":"G000280","Lance Gooden":"G000589","Bob Goodlatte":"G000289","William F. Goodling":"G000291","Carte Patrick Goodwin":"G000561","Bart Gordon":"G000309","Jr. Gore, Albert":"G000321","Slade Gorton":"G000333","Paul A. Gosar":"G000565","Porter J. Goss":"G000336","Josh Gottheimer":"G000583","Trey Gowdy":"G000566","Jr. Gradison, Willis D.":"G000349","Bob Graham":"G000352","Gwen Graham":"G000575","Lindsey Graham":"G000359","Phil Gramm":"G000365","Rod Grams":"G000367","Fred Grandy":"G000371","Kay Granger":"G000377","Bill Grant":"G000382","Chuck Grassley":"G000386","Ella T. Grasso":"G000387","Mike Gravel":"G000388","Garret Graves":"G000577","Sam Graves":"G000546","Tom Graves":"G000560","Kenneth J. Gray":"G000400","III Gray, William H.":"G000402","Alan Grayson":"G000556","Al Green":"G000553","Edith Green":"G000407","Gene Green":"G000410","Mark Green":"G000545","Mark E. Green":"G000590","S. William Green":"G000417","William J. Green":"G000420","James C. Greenwood":"G000439","Judd Gregg":"G000445","Robert P. Griffin":"G000465","Tim Griffin":"G000567","H. Morgan Griffith":"G000568","Parker Griffith":"G000557","Martha W. Griffiths":"G000471","Raul M. Grijalva":"G000551","Michael G. Grimm":"G000569","Wayne R. Grisham":"G000480","H. R. Gross":"G000495","John E. Grotberg":"G000499","Glenn Grothman":"G000576","Jr. Grover, James R.":"G000504","Jr. Grucci, Felix J.":"G000547","Frank J. Guarini":"G000511","Charles S. Gubser":"G000512","Gilbert Gude":"G000513","V. Lamar Gudger":"G000515","Michael Guest":"G000591","Frank C. Guinta":"G000570","Steve Gunderson":"G000524","Bill Gunter":"G000528","Edward J. Gurney":"G000531","Brett Guthrie":"G000558","Luis V. Gutierrez":"G000535","Gil Gutknecht":"G000536","Tennyson Guyer":"G000537","Debra A. Haaland":"H001080","Kay R. Hagan":"H001049","Jim Hagedorn":"H001088","Thomas M. Hagedorn":"H000012","Chuck Hagel":"H001028","Janice Hahn":"H001063","James A. Haley":"H000041","John J. Hall":"H001039","Katie Hall":"H000058","Ralph M. Hall":"H000067","Jr. Hall, Sam B.":"H000070","Tim L. Hall":"H000073","Tony P. Hall":"H000074","Deborah L. Halvorson":"H001044","Dan Hamburg":"H000096","Lee H. Hamilton":"H000114","John P. Hammerschmidt":"H000124","Colleen Hanabusa":"H001050","Kent R. Hance":"H000144","Mel Hancock":"H000151","Karen C. Handel":"H001078","James M. Hanley":"H000158","Richard L. Hanna":"H001051","Richard T. Hanna":"H000164","Mark W. Hannaford":"H000166","Robert P. Hanrahan":"H000168","Clifford P. Hansen":"H000170","George V. Hansen":"H000171","James V. Hansen":"H000172","Julia Butler Hansen":"H000174","Orval Hansen":"H000175","Josh Harder":"H001090","Cresent Hardy":"H001070","Phil Hare":"H001040","Tom Harkin":"H000206","Jane Harman":"H000213","Gregg Harper":"H001045","Michael J. Harrington":"H000230","Andy Harris":"H001052","Claude Harris":"H000236","II Harris, Herbert E.":"H000241","Kamala D. Harris":"H001075","Katherine Harris":"H001035","Frank Harrison":"H000269","William H. Harsha":"H000281","Gary W. Hart":"H000287","Melissa A. Hart":"H001033","Philip A. Hart":"H000291","Vance Hartke":"H000297","Thomas F. Hartnett":"H000302","Vicky Hartzler":"H001053","James Harvey":"H000306","Floyd K. Haskell":"H000317","Margaret Wood Hassan":"H001076","J. Dennis Hastert":"H000323","Alcee L. Hastings":"H000324","Doc Hastings":"H000329","James F. Hastings":"H000327","Orrin G. Hatch":"H000338","Charles Hatcher":"H000340","Mark O. Hatfield":"H000343","Paul Hatfield":"H000344","William D. Hathaway":"H000346","Augustus F. Hawkins":"H000367","Paula Hawkins":"H000374","Josh Hawley":"H001089","Samuel Ichiye Hayakawa":"H000384","Charles A. Hayes":"H000388","Jahana Hayes":"H001081","James A. Hayes":"H000390","Philip H. Hayes":"H000392","Robin Hayes":"H001029","Wayne L. Hays":"H000408","J. D. Hayworth":"H000413","Nan A. S. Hayworth":"H001054","F. Edward Hebert":"H000437","Ken Hechler":"H000438","Chic Hecht":"H000439","Denny Heck":"H001064","Joseph J. Heck":"H001055","Margaret M. Heckler":"H000440","Joel Hefley":"H000444","Howell Heflin":"H000445","W. G. (Bill) Hefner":"H000448","Cecil Heftel":"H000449","Frederick K. (Fred) Heineman":"H000452","Martin Heinrich":"H001046","John Heinz":"H000456","Heidi Heitkamp":"H001069","Dean Heller":"H001041","Jesse Helms":"H000463","Henry Helstoski":"H000465","David N. Henderson":"H000479","Bill Hendon":"H000490","Paul B. Henry":"H000514","Jeb Hensarling":"H001036","Wally Herger":"H000528","Kevin Hern":"H001082","Jaime Herrera Beutler":"H001056","Stephanie Herseth Sandlin":"H001037","Dennis M. Hertel":"H000547","Jody B. Hice":"H001071","Floyd V. Hicks":"H000563","Brian Higgins":"H001038","Clay Higgins":"H001077","Jack E. Hightower":"H000582","John Patrick Hiler":"H000586","Baron P. Hill":"H001030","J. French Hill":"H001072","Katie Hill":"H001087","Rick Hill":"H000605","Van Hilleary":"H000615","Earl F. Hilliard":"H000621","Elwood H. Hillis":"H000624","James A. Himes":"H001047","Maurice D. Hinchey":"H000627","Ruben Hinojosa":"H000636","Andrew J. Hinshaw":"H000638","Jon C. Hinson":"H000641","Mazie K. Hirono":"H001042","Peter Hoagland":"H000652","David L. Hobson":"H000666","George J. Hochbrueckner":"H000670","Kathleen C. Hochul":"H001062","Paul W. Hodes":"H001043","Jr. Hodges, Kaneaster":"H000675","Joseph M. Hoeffel":"H001031","Peter Hoekstra":"H000676","John Hoeven":"H001061","Lawrence J. Hogan":"H000692","Martin R. Hoke":"H000707","Tim Holden":"H000712","George Holding":"H001065","Chet Holifield":"H000713","Kenneth L. Holland":"H000719","Harold C. Hollenbeck":"H000722","Ernest F. Hollings":"H000725","Trey Hollingsworth":"H001074","Clyde Holloway":"H000729","Marjorie S. Holt":"H000747","Rush Holt":"H001032","Elizabeth Holtzman":"H000752","Michael M. Honda":"H001034","Darlene Hooley":"H000762","Larry J. Hopkins":"H000776","Joan Kelly Horn":"H000788","Kendra S. Horn":"H001083","Stephen Horn":"H000789","Steven Horsford":"H001066","Frank J. Horton":"H000797","Craig Hosmer":"H000802","John N. Hostettler":"H000807","Amo Houghton":"H000814","Chrissy Houlahan":"H001085","James J. Howard":"H000840","Allan Howe":"H000851","Steny H. Hoyer":"H000874","Roman L. Hruska":"H000875","Jr. Hubbard, Carroll":"H000878","Robert J. Huber":"H000897","Thomas J. (Jerry) Huckaby":"H000901","Walter (Dee) Huddleston":"H000905","III Hudnut, William H.":"H000906","Richard Hudson":"H001067","Tim Huelskamp":"H001057","Michael Huffington":"H000912","Jared Huffman":"H001068","Harold E. Hughes":"H000922","William J. Hughes":"H000930","Bill Huizenga":"H001058","Kenny C. Hulshof":"H000948","Randy Hultgren":"H001059","Gordon J. Humphrey":"H000951","Hubert H. Humphrey":"H000953","Muriel Humphrey":"H000956","William L. Hungate":"H000965","John E. Hunt":"H000972","Duncan Hunter":"H000981","Duncan D. Hunter":"H001048","Will Hurd":"H001073","Robert Hurt":"H001060","Asa Hutchinson":"H001014","Edward Hutchinson":"H001011","John G. Hutchinson":"H001013","Tim Hutchinson":"H001015","Kay Bailey Hutchison":"H001016","Earl Hutto":"H001018","Henry J. Hyde":"H001022","Cindy Hyde-Smith":"H001079","Richard H. Ichord":"I000001","Bob Inglis":"I000023","James M. Inhofe":"I000024","Daniel K. Inouye":"I000025","Jay Inslee":"I000026","Andrew P. Ireland":"I000029","Johnny Isakson":"I000055","Steve Israel":"I000057","Darrell E. Issa":"I000056","Jr. Istook, Ernest J.":"I000047","Sheila Jackson Lee":"J000032","Henry M. Jackson":"J000013","Jr. Jackson, Jesse L.":"J000283","Jr. Jacobs, Andrew":"J000033","Craig T. James":"J000047","William J. Janklow":"J000286","John Jarman":"J000057","Jacob K. Javits":"J000064","Pramila Jayapal":"J000298","William J. Jefferson":"J000070","James M. Jeffords":"J000072","Hakeem S. Jeffries":"J000294","Jim Jeffries":"J000074","Edgar L. Jenkins":"J000083","Evan H. Jenkins":"J000297","Lynn Jenkins":"J000290","William L. Jenkins":"J000082","Jr. Jenrette, John W.":"J000099","Roger W. Jepsen":"J000101","Bobby Jindal":"J000287","Mike Johanns":"J000291","Christopher John":"J000110","Albert W. Johnson":"J000115","Bill Johnson":"J000292","Don Johnson":"J000120","Dusty Johnson":"J000301","Eddie Bernice Johnson":"J000126","Harold T. Johnson":"J000135","Henry C. \"Hank,\" Jr. Johnson":"J000288","James P. (Jim) Johnson":"J000148","Jay W. Johnson":"J000149","Mike Johnson":"J000299","Nancy L. Johnson":"J000163","Ron Johnson":"J000293","Sam Johnson":"J000174","Tim Johnson":"J000177","Timothy V. Johnson":"J000285","Eugene (Gene) Johnston":"J000200","Harry Johnston":"J000187","J. Bennett Johnston":"J000189","David W. Jolly":"J000296","Ben Jones":"J000211","Brenda Jones":"J000303","Doug Jones":"J000300","Ed Jones":"J000216","James R. Jones":"J000232","Jr. Jones, Robert E.":"J000248","Stephanie Tubbs Jones":"J000284","Walter B. Jones":"J000256","Jr. Jones, Walter B.":"J000255","Jim Jontz":"J000265","Barbara Jordan":"J000266","Jim Jordan":"J000289","David P. Joyce":"J000295","John Joyce":"J000302","Steve Kagen":"K000365","Tim Kaine":"K000384","Paul E. Kanjorski":"K000008","Marcy Kaptur":"K000009","David K. Karnes":"K000011","Joseph E. Karth":"K000014","John R. Kasich":"K000016","Nancy Landon Kassebaum":"K000017","Jr. Kasten, Robert W.":"K000019","Robert W. Kastenmeier":"K000020","John Katko":"K000386","Edward E. Kaufman":"K000373","Jr. Kazen, Abraham":"K000025","William J. Keating":"K000037","William R. Keating":"K000375","Ric Keller":"K000361","Mike Kelly":"K000376","Richard Kelly":"K000077","Robin L. Kelly":"K000385","Sue W. Kelly":"K000078","Trent Kelly":"K000388","Jack Kemp":"K000086","Dirk Kempthorne":"K000088","Edward M. Kennedy":"K000105","John Kennedy":"K000393","II Kennedy, Joseph P.":"K000110","III Kennedy, Joseph P.":"K000379","Mark R. Kennedy":"K000358","Patrick J. Kennedy":"K000113","Barbara B. Kennelly":"K000118","Brian D. Kerns":"K000359","J. Robert Kerrey":"K000146","John F. Kerry":"K000148","William M. Ketchum":"K000153","Martha E. Keys":"K000162","Ro Khanna":"K000389","Ruben J. Kihuen":"K000390","Dale E. Kildee":"K000172","Daniel T. Kildee":"K000380","Derek Kilmer":"K000381","Carolyn C. Kilpatrick":"K000180","Mary Jo Kilroy":"K000372","Andy Kim":"K000394","Jay Kim":"K000181","Ron Kind":"K000188","Thomas N. Kindness":"K000190","Jr. King, Angus S.":"K000383","Carleton J. King":"K000195","Peter T. King":"K000210","Steve King":"K000362","Jack Kingston":"K000220","Adam Kinzinger":"K000378","Mark Steven Kirk":"K000360","Jr. Kirk, Paul Grattan":"K000374","Ann Kirkpatrick":"K000368","Larry Kissell":"K000369","Gerald D. Kleczka":"K000259","Herbert C. Klein":"K000262","Ron Klein":"K000366","John Kline":"K000363","Ron Klink":"K000270","Amy Klobuchar":"K000367","John C. Kluczynski":"K000273","Scott L. Klug":"K000274","Stephen Knight":"K000387","Joe Knollenberg":"K000288","Edward I. Koch":"K000302","Ray Kogovsek":"K000304","Herb Kohl":"K000305","Jim Kolbe":"K000306","Joseph P. Kolter":"K000307","Ernest L. Konnyu":"K000309","Mike Kopetski":"K000312","Suzanne M. Kosmas":"K000370","Peter H. Kostmayer":"K000319","Ken Kramer":"K000322","Jr. Kratovil, Frank":"K000371","John H. Krebs":"K000325","Mike Kreidler":"K000328","Raja Krishnamoorthi":"K000391","Robert C. Krueger":"K000333","Dennis J. Kucinich":"K000336","Jr. Kuhl, John R. \"Randy\"":"K000364","Ann M. Kuster":"K000382","David Kustoff":"K000392","Dan H. Kuykendall":"K000348","Steven T. Kuykendall":"K000357","Jon Kyl":"K000352","Peter N. Kyros":"K000356","Raul R. Labrador":"L000573","John J. LaFalce":"L000556","Robert J. Lagomarsino":"L000020","Darin LaHood":"L000585","Ray LaHood":"L000552","Doug LaMalfa":"L000578","Conor Lamb":"L000588","Doug Lamborn":"L000564","Nick Lampson":"L000043","H. Martin Lancaster":"L000045","Leonard Lance":"L000567","Earl F. Landgrebe":"L000049","Mary L. Landrieu":"L000550","Phil M. Landrum":"L000054","Jeffrey M. Landry":"L000574","James R. Langevin":"L000559","James Lankford":"L000575","Tom Lantos":"L000090","Steve Largent":"L000096","Larry LaRocco":"L000098","Rick Larsen":"L000560","John B. Larson":"L000557","Tom Latham":"L000111","Steven C. LaTourette":"L000553","Delbert L. Latta":"L000116","Robert E. Latta":"L000566","Greg Laughlin":"L000119","Frank R. Lautenberg":"L000123","Brenda L. Lawrence":"L000581","Jr. Lawson, Al":"L000586","Paul D. Laxalt":"L000148","Rick Lazio":"L000155","Jr. Leach, Claude (Buddy)":"L000167","James A. Leach":"L000169","Patrick J. Leahy":"L000174","Marvin Leath":"L000180","John LeBoutillier":"L000547","Raymond F. Lederer":"L000187","Barbara Lee":"L000551","Christopher J. Lee":"L000568","Gary A. Lee":"L000192","Mike Lee":"L000577","Susie Lee":"L000590","Joseph A. LeFante":"L000561","Robert L. Leggett":"L000221","Richard H. Lehman":"L000225","William Lehman":"L000226","Mickey Leland":"L000237","George S. LeMieux":"L000572","Norman F. Lent":"L000243","Debbie Lesko":"L000589","Andy Levin":"L000592","Carl Levin":"L000261","Mike Levin":"L000593","Sander M. Levin":"L000263","Mel Levine":"L000264","Elliott H. Levitas":"L000265","David A. Levy":"L000267","Jason Lewis":"L000587","Jerry Lewis":"L000274","John Lewis":"L000287","Ron Lewis":"L000293","Thomas F. Lewis":"L000295","Joseph I. Lieberman":"L000304","Ted Lieu":"L000582","Jim Lightfoot":"L000305","Blanche L. Lincoln":"L000035","John Linder":"L000321","Daniel Lipinski":"L000563","William O. Lipinski":"L000342","Jerry Litton":"L000360","Bob Livingston":"L000371","James F. Lloyd":"L000379","Marilyn Lloyd":"L000381","Frank A. LoBiondo":"L000554","David Loebsack":"L000565","Tom Loeffler":"L000396","Zoe Lofgren":"L000397","Billy Long":"L000576","Cathy (Mrs. Gillis) Long":"L000411","Clarence D. Long":"L000413","Gillis W. Long":"L000417","Jill L. Long":"L000420","Russell B. Long":"L000428","Jr. Longley, James B.":"L000431","Trent Lott":"L000447","Barry Loudermilk":"L000583","Mia B. Love":"L000584","Alan S. Lowenthal":"L000579","Bill Lowery":"L000479","Nita M. Lowey":"L000480","Mike Lowry":"L000486","Frank D. Lucas":"L000491","Ken Lucas":"L000558","Blaine Luetkemeyer":"L000569","Richard G. Lugar":"L000504","Michelle Lujan Grisham":"L000580","Ben Ray Lujan":"L000570","Jr. Lujan, Manuel":"L000506","Charles Luken":"L000507","Thomas A. Luken":"L000508","Donald E. Lukens":"L000509","Cynthia M. Lummis":"L000571","Stanley N. Lundine":"L000516","Daniel E. Lungren":"L000517","Elaine G. Luria":"L000591","Bill Luther":"L000521","Stephen F. Lynch":"L000562","Thomas MacArthur":"M001193","Torbert H. Macdonald":"M000005","Ronald K. Machtley":"M000015","Connie Mack":"M001155","III Mack, Connie":"M000019","Buddy MacKay":"M000023","Ray J. Madden":"M000039","Edward R. Madigan":"M000041","Daniel B. Maffei":"M001171","Warren G. Magnuson":"M000053","Andrew Maguire":"M000058","George H. Mahon":"M000065","Tim Mahoney":"M001164","William S. Mailliard":"M000070","Denise L. Majette":"M001145","Tom Malinowski":"M001203","Richard W. Mallary":"M000078","Carolyn B. Maloney":"M000087","James H. Maloney":"M000090","Sean Patrick Maloney":"M001185","III Manchin, Joe":"M001183","David Mann":"M000100","James R. Mann":"M000105","Mike Mansfield":"M000113","Thomas J. Manton":"M000117","Donald A. Manzullo":"M001138","Joseph J. Maraziti":"M000121","Kenny Marchant":"M001158","Marjorie Margolies-Mezvinsky":"M000129","Tom Marino":"M001179","Betsy Markey":"M001172","Edward J. Markey":"M000133","Marc L. Marks":"M000136","Ron Marlenee":"M000139","David Daniel Marriott":"M000143","Jim Marshall":"M001146","Roger W. Marshall":"M001198","David O'B. Martin":"M000174","David T. Martin":"M000175","James G. Martin":"M000183","Lynn M. Martin":"M000195","Matthew G. Martinez":"M000206","Mel Martinez":"M001162","William J. Martini":"M000207","Frank Mascara":"M000212","Eric J. J. Massa":"M001173","Thomas Massie":"M001184","Brian J. Mast":"M001199","Jim Matheson":"M001142","Harlan Mathews":"M000236","Jr. Mathias, Charles McC.":"M000241","Robert B. (Bob) Mathias":"M000242","Dawson Mathis":"M000244","Doris O. Matsui":"M001163","Robert T. Matsui":"M000249","Spark M. Matsunaga":"M000250","Mack Mattingly":"M000257","James A. Mattox":"M000260","Nicholas Mavroules":"M000264","Wiley Mayne":"M000286","Romano L. Mazzoli":"M000291","Ben McAdams":"M001209","Vance M. McAllister":"M001192","Lucy McBath":"M001208","John McCain":"M000303","Alfred A. (Al) McCandless":"M000306","Carolyn McCarthy":"M000309","Karen McCarthy":"M000316","Kevin McCarthy":"M001165","Claire McCaskill":"M001170","Michael T. McCaul":"M001157","John L. McClellan":"M000332","Tom McClintock":"M001177","Robert McClory":"M000340","Frank McCloskey":"M000342","Jr. McCloskey, Paul N.":"M000343","James A. McClure":"M000346","John Y. McCollister":"M000349","Betty McCollum":"M001143","Bill McCollum":"M000350","Mitch McConnell":"M000355","Mike McCormack":"M000365","Thaddeus G. McCotter":"M001147","Jim McCrery":"M000388","Dave McCurdy":"M000398","Joseph M. McDade":"M000399","Jim McDermott":"M000404","Lawrence P. McDonald":"M000413","A. Donald McEachin":"M001200","Bob McEwen":"M000432","Robert C. McEwen":"M000433","John J. McFall":"M000436","Gale W. McGee":"M000445","George McGovern":"M000452","James P. McGovern":"M000312","Raymond J. McGrath":"M000458","Paul McHale":"M000466","Patrick T. McHenry":"M001156","John M. McHugh":"M000472","Matthew F. McHugh":"M000473","Scott McInnis":"M000477","David M. McIntosh":"M000481","Mike McIntyre":"M000485","Thomas J. McIntyre":"M000486","K. Gunn McKay":"M000490","Howard P. \"Buck\" McKeon":"M000508","Jr. McKernan, John R.":"M000512","David B. McKinley":"M001180","Cynthia A. McKinney":"M000523","Stewart B. McKinney":"M000527","Michael E. McMahon":"M001174","J. Alex McMillan":"M000566","Thomas McMillen":"M000573","Cathy McMorris Rodgers":"M001159","Jerry McNerney":"M001166","Jr. McNulty, James F.":"M000589","Michael R. McNulty":"M000590","Martha McSally":"M001197","Clem Rogers McSpadden":"M000603","Mark Meadows":"M001187","Lloyd Meeds":"M000626","Martin T. Meehan":"M000627","Patrick Meehan":"M001181","Carrie P. Meek":"M000628","Kendrick B. Meek":"M001148","Gregory W. Meeks":"M001137","Charlie Melancon":"M001161","John Melcher":"M000635","Robert Menendez":"M000639","Grace Meng":"M001188","Jeff Merkley":"M001176","Luke Messer":"M001189","Jack Metcalf":"M000669","Lee Metcalf":"M000671","Ralph H. Metcalfe":"M000675","Howard M. Metzenbaum":"M000678","Daniel Meuser":"M001204","Jan Meyers":"M000684","Helen S. Meyner":"M000685","Edward Mezvinsky":"M000686","Kweisi Mfume":"M000687","Daniel Andrew Mica":"M000688","John L. Mica":"M000689","Michael H. Michaud":"M001149","Robert H. Michel":"M000692","Barbara A. Mikulski":"M000702","Abner J. Mikva":"M000703","Dale Milford":"M000708","Juanita Millender-McDonald":"M000714","Brad Miller":"M001154","Candice S. Miller":"M001150","Carol D. Miller":"M001205","Clarence E. Miller":"M000718","Dan Miller":"M000720","Gary G. Miller":"M001139","George Miller":"M000725","Jeff Miller":"M001144","John R. Miller":"M000736","Zell Miller":"M001141","Wilbur D. Mills":"M000778","William O. Mills":"M000779","Norman Y. Mineta":"M000794","David Minge":"M000795","Joseph G. Minish":"M000796","Patsy T. Mink":"M000797","Walter Minnick":"M001175","William E. Minshall":"M000799","Donald J. Mitchell":"M000808","George J. Mitchell":"M000811","Harry E. Mitchell":"M001167","Parren J. Mitchell":"M000826","Paul Mitchell":"M001201","Wilmer Mizell":"M000833","John Joseph Moakley":"M000834","Toby Moffett":"M000839","Guy Molinari":"M000842","Susan Molinari":"M000843","Alan B. Mollohan":"M000844","Robert H. Mollohan":"M000845","Walter F. Mondale":"M000851","David S. Monson":"M000860","G. V. (Sonny) Montgomery":"M000865","Joseph M. Montoya":"M000876","Jim Moody":"M000881","John R. Moolenaar":"M001194","Alexander X. Mooney":"M001195","Dennis Moore":"M001140","Gwen Moore":"M001160","W. Henson Moore":"M000923","Carlos J. Moorhead":"M000926","William S. Moorhead":"M000930","James P. Moran":"M000933","Jerry Moran":"M000934","Constance A. Morella":"M000941","Joseph D. Morelle":"M001206","Robert B. Morgan":"M000956","Thomas E. Morgan":"M000958","Bruce A. Morrison":"M000992","Sid Morrison":"M000999","Carol Moseley-Braun":"M001025","Charles A. Mosher":"M001031","Frank E. Moss":"M001033","John E. Moss":"M001035","Ronald M. Mottl":"M001044","Seth Moulton":"M001196","Daniel Patrick Moynihan":"M001054","Robert J. Mrazek":"M001057","Debbie Mucarsel-Powell":"M001207","Markwayne Mullin":"M001190","Mick Mulvaney":"M001182","Frank H. Murkowski":"M001085","Lisa Murkowski":"M001153","Austin J. Murphy":"M001088","Christopher Murphy":"M001169","John M. Murphy":"M001098","Morgan F. Murphy":"M001101","Patrick Murphy":"M001191","Patrick J. Murphy":"M001168","Scott Murphy":"M001178","Stephanie N. Murphy":"M001202","Tim Murphy":"M001151","Patty Murray":"M001111","John P. Murtha":"M001120","Marilyn N. Musgrave":"M001152","Edmund S. Muskie":"M001121","Raphael Musto":"M001123","Gary A. Myers":"M001128","John T. Myers":"M001130","Michael O. Myers":"M001132","Sue Wilkins Myrick":"M001134","Jerrold Nadler":"N000002","Dave R, Nagle":"N000003","John L. Napier":"N000005","Grace F. Napolitano":"N000179","William H. Natcher":"N000009","Richard E. Neal":"N000015","Stephen L. Neal":"N000016","Lucien N. Nedzi":"N000019","Gloria Negrete McLeod":"N000187","Joe Neguse":"N000191","James L. Nelligan":"N000027","Ancher Nelsen":"N000028","Ben Nelson":"N000180","Bill Nelson":"N000032","Gaylord Nelson":"N000033","Jr. Nethercutt, George R.":"N000051","Randy Neugebauer":"N000182","Mark W. Neumann":"N000054","Dan Newhouse":"N000189","Robert W. Ney":"N000081","Bill Nichols":"N000095","Dick Nichols":"N000094","Don Nickles":"N000102","Howard C. Nielson":"N000106","Robert N. C. Nix":"N000113","Kristi L. Noem":"N000184","Richard M. Nolan":"N000127","Donald Norcross":"N000188","Ralph Norman":"N000190","Anne M. Northup":"N000143","Eleanor Holmes Norton":"N000147","Charles W. Norwood":"N000159","Henry Nowak":"N000163","Richard B. Nugent":"N000185","Devin Nunes":"N000181","Sam Nunn":"N000171","Alan Nunnelee":"N000186","Jim Nussle":"N000172","Glenn C. Nye":"N000183","George M. O'Brien":"O000010","Tom O'Halleran":"O000171","James G. O'Hara":"O000055","Jr. O'Neill, Thomas P.":"O000098","Beto O'Rourke":"O000170","Mary Rose Oakar":"O000001","Barack Obama":"O000167","James L. Oberstar":"O000006","David R. Obey":"O000007","Alexandria Ocasio-Cortez":"O000172","James R. Olin":"O000069","Pete Olson":"O000168","John W. Olver":"O000085","Ilhan Omar":"O000173","Solomon P. Ortiz":"O000107","Bill Orton":"O000108","Tom Osborne":"O000165","Doug Ose":"O000164","C. L. (Butch) Otter":"O000166","Richard L. Ottinger":"O000134","Major R. Owens":"O000159","Wayne Owens":"O000156","William L. Owens":"O000169","Michael G. Oxley":"O000163","Ron Packard":"P000005","Bob Packwood":"P000009","Steven M. Palazzo":"P000601","Jr. Pallone, Frank":"P000034","Gary J. Palmer":"P000609","Jimmy Panetta":"P000613","Leon Panetta":"P000047","Chris Pappas":"P000614","Michael Pappas":"P000049","Mike Parker":"P000066","Stanford E. (Stan) Parris":"P000080","Jr. Pascrell, Bill":"P000096","Jr. Pashayan, Charles":"P000097","Otto E. Passman":"P000098","Ed Pastor":"P000099","John O. Pastore":"P000100","William N. Patman":"P000104","Wright Patman":"P000103","Edward J. Patten":"P000106","Elizabeth Patterson":"P000112","Jerry M. Patterson":"P000121","Edward W. Pattison":"P000134","Rand Paul":"P000603","Ron Paul":"P000583","Erik Paulsen":"P000594","Bill Paxon":"P000148","Donald M. Payne":"P000149","Jr. Payne, Donald M.":"P000604","Jr. Payne, Lewis F.":"P000152","Stevan Pearce":"P000588","James B. Pearson":"P000166","Donald J. Pease":"P000170","Edward A. Pease":"P000171","Claiborne Pell":"P000193","Nancy Pelosi":"P000197","Greg Pence":"P000615","Mike Pence":"P000587","Timothy J. Penny":"P000215","Claude Pepper":"P000218","Charles H. Percy":"P000222","David Perdue":"P000612","Carl C. Perkins":"P000229","Carl Dewey Perkins":"P000230","Ed Perlmutter":"P000593","Thomas S.P. Perriello":"P000600","Scott Perry":"P000605","Gary C. Peters":"P000595","Scott H. Peters":"P000608","Collin C. Peterson":"P000258","Douglas (Pete) Peterson":"P000259","John E. Peterson":"P000263","Thomas E. Petri":"P000265","Jerry L. Pettis":"P000272","Shirley N. Pettis":"P000273","Peter A. Peyser":"P000280","David D. Phelps":"P000584","Dean Phillips":"P000616","Charles W. \"Chip\" Pickering":"P000323","Owen B. Pickett":"P000326","J. J. Pickle":"P000328","Pedro R. Pierluisi":"P000596","Otis G. Pike":"P000348","Chellie Pingree":"P000597","Robert Pittenger":"P000606","Joseph R. Pitts":"P000373","Stacey E. Plaskett":"P000610","Todd Russell Platts":"P000585","W. R. Poage":"P000398","Mark Pocan":"P000607","Bertram L. Podell":"P000399","Ted Poe":"P000592","Bruce Poliquin":"P000611","Jared Polis":"P000598","Richard W. Pombo":"P000419","Earl Pomeroy":"P000422","Mike Pompeo":"P000602","John Edward Porter":"P000444","Jon C. Porter":"P000589","Katie Porter":"P000618","Rob Portman":"P000449","Bill Posey":"P000599","Glenn Poshard":"P000452","Walter E. Powell":"P000485","Larry Pressler":"P000513","Ayanna Pressley":"P000617","Richardson Preyer":"P000520","David E. Price":"P000523","Melvin Price":"P000522","Robert D. Price":"P000528","Tom Price":"P000591","Joel Pritchard":"P000546","William Proxmire":"P000553","Deborah Pryce":"P000555","David H. Pryor":"P000556","Mark L. Pryor":"P000590","Carl D. Pursell":"P000574","Adam H. Putnam":"P000586","Benjamin Quayle":"Q000024","Dan Quayle":"Q000007","Albert H. Quie":"Q000010","Mike Quigley":"Q000023","James H. (Jimmy) Quillen":"Q000013","Jack Quinn":"Q000016","George Radanovich":"R000004","Trey Radel":"R000596","Aumua Amata Coleman Radewagen":"R000600","II Rahall, Nick J.":"R000011","Thomas F. Railsback":"R000012","Jim Ramstad":"R000033","William J. Randall":"R000041","Jennings Randolph":"R000046","Charles B. Rangel":"R000053","John R. Rarick":"R000065","Jamie Raskin":"R000606","William R. Ratchford":"R000067","John Ratcliffe":"R000601","Jr. Ravenel, Arthur":"R000072","Richard B. Ray":"R000080","Bill Redmond":"R000567","Jack Reed":"R000122","Tom Reed":"R000585","Thomas N. Rees":"R000134","Ralph Regula":"R000141","Denny Rehberg":"R000571","David G. Reichert":"R000578","Harry Reid":"R000146","Ogden R. Reid":"R000150","James B. Renacci":"R000586","Rick Renzi":"R000574","Guy Reschenthaler":"R000610","Henry S. Reuss":"R000165","Silvestre Reyes":"R000170","Mel Reynolds":"R000178","Thomas M. Reynolds":"R000569","John J. Rhodes":"R000188","III Rhodes, John J.":"R000189","Reid J. Ribble":"R000587","Abraham A. Ribicoff":"R000191","Kathleen M. Rice":"R000602","Tom Rice":"R000597","Bill Richardson":"R000229","Laura Richardson":"R000581","Cedric L. Richmond":"R000588","Frederick W. Richmond":"R000232","Thomas J. Ridge":"R000243","Jr. Riegle, Donald W.":"R000249","E. Scott Rigell":"R000589","Denver Riggleman":"R000611","Frank Riggs":"R000252","Bob Riley":"R000258","Matthew J. Rinaldo":"R000262","James E. Risch":"R000584","Theodore M. Risenhoover":"R000268","Don Ritter":"R000277","David Rivera":"R000590","Lynn N. Rivers":"R000281","Charles S. Robb":"R000295","Clint Roberts":"R000308","Pat Roberts":"R000307","Ray Roberts":"R000312","J. Kenneth Robinson":"R000337","Tommy F. Robinson":"R000354","Howard W. Robison":"R000357","Martha Roby":"R000591","IV Rockefeller, John D.":"R000361","Jr. Rodino, Peter W.":"R000374","Ciro D. Rodriguez":"R000568","David P. Roe":"R000582","Robert A. Roe":"R000383","Buddy Roemer":"R000384","Tim Roemer":"R000385","James E. Rogan":"R000386","Harold Rogers":"R000395","Mike D. Rogers":"R000575","Mike J. Rogers":"R000572","Paul G. Rogers":"R000401","Dana Rohrabacher":"R000409","Todd Rokita":"R000592","Carlos A. Romero-Barcelo":"R000417","Mitt Romney":"R000615","Teno Roncalio":"R000421","Angelo D. Roncallo":"R000422","Francis Rooney":"R000607","Fred B. Rooney":"R000423","John J. Rooney":"R000424","Thomas J. Rooney":"R000583","Ileana Ros-Lehtinen":"R000435","Charlie Rose":"R000436","John W. Rose":"R000612","Max Rose":"R000613","Jacky Rosen":"R000608","Benjamin S. Rosenthal":"R000442","Peter J. Roskam":"R000580","Dennis A. Ross":"R000593","Mike Ross":"R000573","Dan Rostenkowski":"R000458","William V. Roth Jr.":"R000460","Toby Roth":"R000459","Keith J. Rothfus":"R000598","Steven R. Rothman":"R000462","Harley Rouda":"R000616","Marge Roukema":"R000465","Mike Rounds":"R000605","J. Edward Roush":"R000467","John H. Rousselot":"R000469","David Rouzer":"R000603","J. Roy Rowland":"R000481","John G. Rowland":"R000482","Chip Roy":"R000614","William R. Roy":"R000484","Edward R. Roybal":"R000485","Lucille Roybal-Allard":"R000486","Edward R. Royce":"R000487","William Royer":"R000489","Marco Rubio":"R000595","Eldon D. Rudd":"R000495","Warren Rudman":"R000497","Raul Ruiz":"R000599","Harold L. Runnels":"R000510","Jon Runyan":"R000594","Philip E. Ruppe":"R000512","C. A. Dutch Ruppersberger":"R000576","Bobby L. Rush":"R000515","Steve Russell":"R000604","Martin A. Russo":"R000543","Earl B. Ruth":"R000545","John H. Rutherford":"R000609","Leo J. Ryan":"R000558","Paul D. Ryan":"R000570","Tim Ryan":"R000577","Jim Ryun":"R000566","Gregorio Kilili Camacho Sablan":"S001177","Martin Olav Sabo":"S000005","Patricia Saiki":"S000014","John T. Salazar":"S001158","Ken Salazar":"S001163","Bill Sali":"S001167","Matt Salmon":"S000018","Michael F. Q. San Nicolas":"S001204","Linda T. Sanchez":"S001156","Loretta Sanchez":"S000030","Bernard Sanders":"S000033","Max Sandlin":"S000044","Jr. Sandman, Charles W.":"S000045","Mark Sanford":"S000051","Terry Sanford":"S000055","George E. Sangmeister":"S000056","James D. Santini":"S000058","Rick Santorum":"S000059","Ronald A. Sarasin":"S000062","John P. Sarbanes":"S001168","Paul S. Sarbanes":"S000064","Bill Sarpalius":"S000066","Ben Sasse":"S001197","Jim Sasser":"S000068","III Satterfield, David E.":"S000070","Gus Savage":"S000081","Harold S. Sawyer":"S000087","Tom Sawyer":"S000094","William B. Saxbe":"S000096","Jim Saxton":"S000097","John P. Saylor":"S000102","Steve Scalise":"S001176","Mary Gay Scanlon":"S001205","Joe Scarborough":"S000106","Dan Schaefer":"S000109","Bob Schaffer":"S000112","Janice D. Schakowsky":"S001145","Brian Schatz":"S001194","Mark H. Schauer":"S001178","Lynn Schenk":"S000119","William J. Scherle":"S000121","James H. Scheuer":"S000124","Adam B. Schiff":"S001150","Steven Schiff":"S000125","Robert T. Schilling":"S001182","Jean Schmidt":"S001164","Harrison H. Schmitt":"S000132","Herman T. Schneebeli":"S000135","Bradley Scott Schneider":"S001190","Claudine Schneider":"S000136","Aaron Schock":"S001179","Kurt Schrader":"S001180","Kim Schrier":"S001216","Edward L. Schrock":"S001151","Patricia Schroeder":"S000142","Bill Schuette":"S000143","Richard T. Schulze":"S000146","Charles E. Schumer":"S000148","Allyson Y. Schwartz":"S001162","John J.H. \"Joe\" Schwarz":"S001161","Richard S. Schweiker":"S000159","David Schweikert":"S001183","Austin Scott":"S001189","David Scott":"S001157","Hugh Scott":"S000174","Rick Scott":"S001217","Robert C. \"Bobby\" Scott":"S000185","Tim Scott":"S001184","William Lloyd Scott":"S000189","Andrea H. Seastrand":"S000213","Keith G. Sebelius":"S000217","John F. Seiberling":"S000230","Shelley Sekula Gibbs":"S001166","Jr. Sensenbrenner, F. James":"S000244","Jose E. Serrano":"S000248","Jeff Sessions":"S001141","Pete Sessions":"S000250","Joe Sestak":"S001169","Terri A. Sewell":"S001185","John Seymour":"S000269","John B. Shadegg":"S000275","Jeanne Shaheen":"S001181","Donna E. Shalala":"S001206","Bob Shamansky":"S000282","James M. Shannon":"S000286","Philip R. Sharp":"S000294","Jr. Shaw, E. Clay":"S000303","Christopher Shays":"S001144","Carol Shea-Porter":"S001170","Richard C. Shelby":"S000320","Karen Shepherd":"S000332","Brad Sherman":"S000344","Mikie Sherrill":"S001207","Don Sherwood":"S001146","John Shimkus":"S000364","George E. Shipley":"S000367","Dick Shoup":"S000383","Ronnie Shows":"S001147","Garner E. Shriver":"S000388","Heath Shuler":"S001171","Norman D. Shumway":"S000393","Bill Shuster":"S001154","Bud Shuster":"S000394","Robert L. F. Sikes":"S000406","Gerry E. Sikorski":"S000407","Mark D. Siljander":"S000409","Rob Simmons":"S001152","Paul Simon":"S000423","Alan K. Simpson":"S000429","Michael K. Simpson":"S001148","Kyrsten Sinema":"S001191","Albio Sires":"S001165","Norman Sisisky":"S000453","B. F. Sisk":"S000454","David E. Skaggs":"S000462","Joe Skeen":"S000463","Ike Skelton":"S000465","Joe Skubitz":"S000472","John Slack":"S000473","Jim Slattery":"S000477","Jr. Slaughter, D. French":"S000479","Louise McIntosh Slaughter":"S000480","Elissa Slotkin":"S001208","Adam Smith":"S000510","Adrian Smith":"S001172","Jr. Smith, Albert L.":"S000514","Bob Smith":"S000606","Christopher H. Smith":"S000522","Denny Smith":"S000527","Gordon H. Smith":"S001142","III Smith, Henry P.":"S000548","Jason Smith":"S001195","Joseph F. Smith":"S000579","Lamar Smith":"S000583","Larkin Smith":"S000584","Lawrence Smith":"S000586","Linda Smith":"S000587","Neal Edward Smith":"S000596","Nick Smith":"S000597","Peter Smith":"S000601","Robert Smith":"S000607","Tina Smith":"S001203","Virginia Smith":"S000622","Lloyd Smucker":"S001199","Vince Snowbarger":"S000662","Olympia J. Snowe":"S000663","M. G. (Gene) Snyder":"S000669","Vic Snyder":"S000672","Michael E. Sodrel":"S001160","Stephen J. Solarz":"S000673","Hilda L. Solis":"S001153","Gerald B. H. Solomon":"S000675","Darren Soto":"S001200","Mark E. Souder":"S001143","Steve II Southerland":"S001186","Zachary T. Space":"S001173","Abigail Davis Spanberger":"S001209","Ross Spano":"S001210","John J. Sparkman":"S000701","Arlen Specter":"S000709","Jackie Speier":"S001175","Gladys Noon Spellman":"S000716","Floyd Spence":"S000718","Jr. Spratt, John M.":"S000749","Fernand J. St Germain":"S000762","Debbie Stabenow":"S000770","Edward J. Stack":"S000772","Robert T. Stafford":"S000776","Harley O. Staggers":"S000778","Jr. Staggers, Harley O.":"S000779","Richard H. Stallings":"S000785","Arlan Stangeland":"S000795","Greg Stanton":"S001211","J. William Stanton":"S000804","James V. Stanton":"S000803","Fortney Pete Stark":"S000810","David Michael Staton":"S000818","Pete Stauber":"S001212","Cliff Stearns":"S000822","Tom Steed":"S000829","Robert H. Steele":"S000836","Alan Steelman":"S000841","Jr. Steers, Newton I.":"S000844","Elise M. Stefanik":"S001196","Sam Steiger":"S000846","William A. Steiger":"S000847","Bryan Steil":"S001213","Charles W. Stenholm":"S000851","John C. Stennis":"S000852","Jr. Stephens, Robert G.":"S000860","W. Gregory Steube":"S001214","Haley M. Stevens":"S001215","Ted Stevens":"S000888","III Stevenson, Adlai E.":"S000890","Bennett McVey Stewart":"S000902","Chris Stewart":"S001192","Donald Stewart":"S000907","Steve Stivers":"S001187","David A. Stockman":"S000935","Steve Stockman":"S000937","Louis Stokes":"S000948","Richard (Dick) Stone":"S000962","Michael L. Strang":"S000991","Luther Strange":"S001202","Samuel S. Stratton":"S000997","Ted Strickland":"S001004","Frank A. Stubblefield":"S001037","Jr. Stuckey, W. S. (Bill)":"S001039","Gerry E. Studds":"S001040","Bob Stump":"S001044","Bart Stupak":"S001045","Marlin A. Stutzman":"S001188","Dan Sullivan":"S001198","John Sullivan":"S001155","Leonor K. (Mrs. John B.) Sullivan":"S001057","Don Sundquist":"S001075","Foto I. F. Sunia":"S001077","John E. Sununu":"S001078","Thomas R. Suozzi":"S001201","Betty Sutton":"S001174","Eric Swalwell":"S001193","David McCann Sweeney":"S001101","John E. Sweeney":"S001149","Dick Swett":"S001113","Al Swift":"S001115","Patrick L. Swindall":"S001122","James W. Symington":"S001134","Stuart Symington":"S001136","Steven D. Symms":"S001138","Mike Synar":"S001139","Jr. Taft, Robert":"T000010","Mark Takai":"T000473","Mark Takano":"T000472","Burt L. Talcott":"T000022","Jim Talent":"T000024","Robert M. (Robin) Tallon":"T000034","Herman E. Talmadge":"T000035","Thomas G. Tancredo":"T000458","John S. Tanner":"T000038","Randy Tate":"T000048","Thomas Joseph Tauke":"T000053","Ellen O. Tauscher":"T000057","W. J. (Billy) Tauzin":"T000058","Charles H. Taylor":"T000067","G. Taylor":"T000075","Gene Taylor":"T000074","Roy A. Taylor":"T000099","Scott Taylor":"T000477","Van Taylor":"T000479","Charles M. Teague":"T000109","Harry Teague":"T000466","Olin E. Teague":"T000110","Frank Tejeda":"T000113","Claudia Tenney":"T000478","Lee Terry":"T000459","Jon Tester":"T000464","Craig Thomas":"T000162","Lindsay Thomas":"T000184","William M. Thomas":"T000188","Bennie G. Thompson":"T000193","Jr. Thompson, Frank":"T000200","Fred Thompson":"T000457","Glenn Thompson":"T000467","Mike Thompson":"T000460","Vernon W. Thomson":"T000233","Charles Thone":"T000234","Mac Thornberry":"T000238","Ray Thornton":"T000243","John Thune":"T000250","Karen L. Thurman":"T000253","Strom Thurmond":"T000254","Todd Tiahrt":"T000260","Patrick J. Tiberi":"T000462","Robert O. Tiernan":"T000265","John F. Tierney":"T000266","Thom Tillis":"T000476","William R. IV Timmons":"T000480","Scott R. Tipton":"T000470","Dina Titus":"T000468","Rashida Tlaib":"T000481","Paul Tonko":"T000469","Richard A. Tonry":"T000310","Pat Toomey":"T000461","Peter G. Torkildsen":"T000314","Xochitl Torres Small":"T000484","Estaban Edward Torres":"T000316","Norma J. Torres":"T000474","Robert G. Torricelli":"T000317","David Towell":"T000321","John G. Tower":"T000322","Edolphus Towns":"T000326","Jr. Traficant, James A.":"T000350","Lori Trahan":"T000482","Bob Traxler":"T000356","David C. Treen":"T000362","Jr. Trible, Paul S.":"T000367","David J. Trone":"T000483","David A. Trott":"T000475","Niki Tsongas":"T000465","Paul E. Tsongas":"T000393","Jr. (Jim Guy) Tucker, James G.":"T000400","III Tucker, Walter R.":"T000405","John V. Tunney":"T000410","Jim Turner":"T000424","Michael R. Turner":"T000463","Robert L. Turner":"T000471","Mark Udall":"U000038","Morris K. Udall":"U000001","Tom Udall":"U000039","Al Ullman":"U000004","Lauren Underwood":"U000040","Robert A. Underwood":"U000014","Jolene Unsoeld":"U000017","Fred Upton":"U000031","David G. Valadao":"V000129","Tim Valentine":"V000006","Lionel Van Deerlin":"V000024","Jefferson Van Drew":"V000133","Chris Van Hollen":"V000128","Guy Vander Jagt":"V000027","Richard F. Vander Veen":"V000029","Tommy J. Vandergriff":"V000026","Charles A. Vanik":"V000047","Juan Vargas":"V000130","Marc A. Veasey":"V000131","Filemon Vela":"V000132","Nydia M. Velazquez":"V000081","Bruce F. Vento":"V000087","Victor V. Veysey":"V000093","Joseph P. Vigorito":"V000098","Peter J. Visclosky":"V000108","David Vitter":"V000127","George V. Voinovich":"V000126","Harold L. Volkmer":"V000112","Barbara F. Vucanovich":"V000124","Jr. Waggonner, Joe D.":"W000018","Ann Wagner":"W000812","Tim Walberg":"W000798","Greg Walden":"W000791","Enid Greene Waldholtz":"G000408","Jerome R. Waldie":"W000035","Jr. Waldon, Alton R.":"W000038","Doug Walgren":"W000044","Mark Walker":"W000819","Robert S. Walker":"W000068","Malcolm Wallop":"W000092","Jackie Walorski":"W000813","James T. Walsh":"W000099","Joe Walsh":"W000811","John E. Walsh":"W000818","William F. Walsh":"W000107","Mimi Walters":"W000820","Michael Waltz":"W000823","Timothy J. Walz":"W000799","Zach Wamp":"W000119","William C. Wampler":"W000121","Mike Ward":"W000139","John H. Ware":"W000147","John Warner":"W000154","Mark R. Warner":"W000805","Elizabeth Warren":"W000817","Craig A. Washington":"W000177","Harold Washington":"W000180","Debbie Wasserman Schultz":"W000797","Maxine Waters":"W000187","Steve Watkins":"W000824","Wes Watkins":"W000194","Bonnie Watson Coleman":"W000822","Diane E. Watson":"W000794","Melvin L. Watt":"W000207","Jr. Watts, J. C.":"W000210","Henry A. Waxman":"W000215","James H. Weaver":"W000227","Jim Webb":"W000803","Ed Weber":"W000235","Sr. Weber, Randy K.":"W000814","Vin Weber":"W000237","Daniel Webster":"W000806","Jr. Weicker, Lowell P.":"W000253","Anthony D. Weiner":"W000792","Ted Weiss":"W000258","Peter Welch":"W000800","Curt Weldon":"W000268","Dave Weldon":"W000267","Jerry Weller":"W000273","Paul D. Wellstone":"W000288","Brad R. Wenstrup":"W000815","Allen B. West":"W000807","Bruce Westerman":"W000821","Lynn A. Westmoreland":"W000796","Robert Wexler":"W000314","Jennifer Wexton":"W000825","Robert A. Weygand":"W000315","Jr. Whalen, Charles W.":"W000317","Alan Wheat":"W000326","Richard C. White":"W000390","Rick White":"W000391","Sheldon Whitehouse":"W000802","G. William Whitehurst":"W000406","Ed Whitfield":"W000413","Sr. Whitley, Charles O.":"W000419","Bob Whittaker":"W000426","Jamie L. Whitten":"W000428","Roger F. Wicker":"W000437","William B. Widnall":"W000445","Charles E. Wiggins":"W000448","Susan Wild":"W000826","Jr. Williams, Harrison A.":"W000502","Lawrence G. Williams":"W000525","Lyle Williams":"W000528","Pat Williams":"W000520","Roger Williams":"W000816","Charles Wilson":"W000570","Charles A. Wilson":"W000801","Charles H. Wilson":"W000571","Frederica S. Wilson":"W000808","Heather Wilson":"W000789","Joe Wilson":"W000795","Pete Wilson":"W000607","Robert C. Wilson":"W000610","Jr. Winn, Larry":"W000636","Timothy Wirth":"W000647","Jr. Wise, Robert E.":"W000654","Robert J. Wittman":"W000804","Harris Wofford":"W000665","Frank R. Wolf":"W000672","Lester L. Wolff":"W000680","Howard E. Wolpe":"W000682","Steve Womack":"W000809","Antonio B. Won Pat":"W000686","Rob Woodall":"W000810","Lynn C. Woolsey":"W000738","George C. Wortley":"W000752","Jr. Wright, James C.":"W000763","Ron Wright":"W000827","David Wu":"W000793","Jr. Wyatt, Joe":"W000777","Wendell Wyatt":"W000778","Ron Wyden":"W000779","John W. Wydler":"W000780","Chalmers P. Wylie":"W000781","Louis C. Wyman":"W000782","Albert Russell Wynn":"W000784","John A. Yarmuth":"Y000062","Sidney R. Yates":"Y000013","Gus Yatron":"Y000014","Kevin Yoder":"Y000063","Ted S. Yoho":"Y000065","Andrew Young":"Y000028","C. W. Bill Young":"Y000031","David Young":"Y000066","Don Young":"Y000033","Edward Young":"Y000035","John Young":"Y000043","Milton R. Young":"Y000047","Robert A. Young":"Y000051","Samuel H. Young":"Y000052","Todd Young":"Y000064","Clement J. Zablocki":"Z000001","Leo C. Zeferetti":"Z000002","Lee M. Zeldin":"Z000017","Jr. Zeliff, William H.":"Z000004","Dick Zimmer":"Z000008","Ryan K. Zinke":"Z000018","Roger H. Zion":"Z000010","Edward Zorinsky":"Z000013","Edwin V. W. Zschau":"Z000014","John M. Zwach":"Z000016"}
  
    senator1=window.localStorage.getItem('Senator1');
    senator2=window.localStorage.getItem('Senator2');
    rep=window.localStorage.getItem('Rep');
        if (IDs.hasOwnProperty(senator1)) {
          console.log("found a match for: "+ senator1);
          window.localStorage.setItem('S1ID', IDs[senator1]);
        } else {
          console.log("No senator1");
          window.localStorage.setItem('S1ID', 'fail');
        }
        if (IDs.hasOwnProperty(senator2)) {
           console.log("found a match for: " + senator2);
           window.localStorage.setItem('S2ID', IDs[senator2]);
        } else {
          console.log("No senator2");
          window.localStorage.setItem('S2ID', 'fail');
        }
        if (IDs.hasOwnProperty(rep)) {
          console.log("found a match for: " + rep);
          window.localStorage.setItem('RID', IDs[rep]);
        } else {
          console.log("No Rep");
          window.localStorage.setItem('RID', 'fail');
        }
  }
  
  function ooo(f, t) {
    var housecommittees = {
      "Committee on Administration":  "HSHA",
      "Committee on Agriculture": "HSAG",
      "Committee on Appropriations": "HSAP",
      "Committee on Armed Services": "HSAS",
      "Committee on Education and Labor": "HSED",
      "Committee on Energy and Commerce": "HSIF",
      "Committee on Ethics": "HSSO",
      "Committee on Financial Services": "HSBA",
      "Committee on Foreign Affairs":  "HSFA",
      "Committee on Homeland Security":  "HSHM",
      "Committee on Natural Resources": "HSII",
      "Committee on Oversight and Reform":  "HSGO",
      "Committee on Rules":  "HSRU",
      "Committee on Science, Space, and Technology":  "HSSY",
      "Committee on Small Business": "HSSM",
      "Committee on the Budget": "HSBU",
      "Committee on the Judiciary": "HSJU",
      "Committee on Transportation and Infrastructure": "HSPW",
      "Committee on Veterans' Affairs": "HSVR",
      "Committee on Ways and Means": "HSWM",
      "Permanent Select Committee on Intelligence": "HSIG",
      "Select Committee on the Climate Crisis":  "HSCN",
      "Select Committee on the Modernization of Congress": "HSMH",
    };
    var housesub = {
      "HElections": "HSHA01",
      "HElections": "HSHA08",
      "Nutrition, Oversight, and Department Operations": "HSAG03",
      "General Farm Commodities and Risk Management": "HSAG16",
      "Conservation and Forestry": "HSAG15",
      "Commodity Exchanges, Energy, and Credit": "HSAG22",
      "Livestock and Foreign Agriculture": "HSAG29",
      "Biotechnology, Horticulture, and Research": "HSAG14",
      "Agriculture, Rural Development, Food and Drug Administration, and Related Agencies": "HSAP01",
      "Commerce, Justice, Science, and Related Agencies": "HSAP19",
      "Defense": "HSAP02",
      "Energy and Water Development, and Related Agencies": "HSAP10",
      "Financial Services and General Government": "HSAP23",
      "Homeland Security": "HSAP15",
      "Interior, Environment, and Related Agencies": "HSAP06",
      "Labor, Health and Human Services, Education, and Related Agencies": "HSAP07",
      "Legislative Branch": "HSAP24",
      "Military Construction, Veterans Affairs, and Related Agencies": "HSAP18",
      "State, Foreign Operations, and Related Programs": "HSAP04",
      "Transportation, and Housing and Urban Development, and Related Agencies": "HSAP20",
      "Tactical Air and Land Forces": "HSAS25",
      "Military Personnel": "HSAS02",
      "Readiness": "HSAS03",
      "Seapower and Projection Forces": "HSAS28",
      "Strategic Forces": "HSAS29",
      "Intelligence and Emerging Threats and Capabilities": "HSAS26",
      "Early Childhood, Elementary, and Secondary Education": "HSED14",
      "Health, Employment, Labor, and Pensions": "HSED02",
      "Higher Education and Workforce Investment": "HSED13",
      "Workforce Protections": "HSED10",
      "Civil Rights and Human Services": "HSED07",
      "Consumer Protection and Commerce": "HSIF17",
      "Communications and Technology": "HSIF16",
      "Energy": "HSIF03",
      "Health": "HSIF14",
      "Oversight and Investigations": "HSIF02",
      "Environment and Climate Change": "HSIF18",
      "Investor Protection, Entrepreneurship, and Capital Markets": "HSBA16",
      "Consumer Protection and Financial Institutions": "HSBA15",
      "Housing, Community Development, and Insurance": "HSBA04",
      "Oversight and Investigations": "HSBA09",
      "National Security, International Development, and Monetary Policy": "HSBA01",
      "Diversity and Inclusion": "HSBA13",
      "Africa, Global Health, Global Human Rights, and International Organizations": "HSFA16",
      "Asia, the Pacific, and Nonproliferation": "HSFA05",
      "Europe, Eurasia, Energy, and the Environment": "HSFA14",
      "Oversight and Investigations": "HSFA17",
      "Middle East, North Africa, and International Terrorism": "HSFA13",
      "Western Hemisphere, Civilian Security, and Trade": "HSFA07",
      "Energy and Mineral Resources": "HSII06",
      "Fisheries, Wildlife, Oceans and Insular Affairs": "HSII22",
      "National Parks, Forests, and Public Lands": "HSII10",
      "Water, Oceans, and Wildlife": "HSII13",
      "Oversight and Investigations": "HSII15",
      "Indigenous Peoples of the United States": "HSII24",
      "Information Technology": "HSGO25",
      "Government Operations": "HSGO24",
      "National Security": "HSGO06",
      "Environment": "HSGO28",
      "Healthcare, Benefits, and Administrative Rules": "HSGO27",
      "Intergovernmental Affairs": "HSGO29",
      "Civil Rights and Civil Liberties": "HSGO02",
      "Economic and Consumer Policy": "HSGO05",
      "Legislative and Budget Process": "HSRU02",
      "Rules and Organization of the House": "HSRU04",
      "Expedited Procedures": "HSRU05",
      "Energy": "HSSY20",
      "Investigations and Oversight": "HSSY21",
      "Space and Aeronautics": "HSSY16",
      "Research and Technology": "HSSY15",
      "Environment": "HSSY18",
      "Contracting and Infrastructure": "HSSM23",
      "Investigations, Oversight and Regulations": "HSSM24",
      "Rural Development, Agriculture, Trade and Entrepreneurship": "HSSM25",
      "Economic Growth, Tax and Capital Access": "HSSM27",
      "Innovation and Workforce Development": "HSSM26",
      "Contracting and Infrastructure": "HSSM23",
      "Investigations, Oversight and Regulations": "HSSM24",
      "Rural Development, Agriculture, Trade and Entrepreneurship": "HSSM25",
      "Economic Growth, Tax and Capital Access": "HSSM27",
      "Innovation and Workforce Development": "HSSM26",
      "Constitution, Civil Rights, and Civil Justice": "HSJU10",
      "Courts, Intellectual Property, and the Internet": "HSJU03",
      "Crime, Terrorism and Homeland Security": "HSJU08",
      "Immigration and Citizenship": "HSJU01",
      "Antitrust, Commercial and Administrative Law": "HSJU05",
      "Aviation": "HSPW05",
      "Coast Guard and Maritime Transportation": "HSPW07",
      "Economic Development, Public Buildings, and Emergency Management": "HSPW13",
      "Highways and Transit": "HSPW12",
      "Railroads, Pipelines, and Hazardous Materials": "HSPW14",
      "Water Resources and Environment": "HSPW02",
      "Disability Assistance and Memorial Affairs": "HSVR09",
      "Economic Opportunity": "HSVR10",
      "Health": "HSVR03",
      "Oversight and Investigations": "HSVR08",
      "Technology Modernization": "HSVR11",
      "Health": "HSWM02",
      "Worker and Family Support": "HSWM03",
      "Oversight": "HSWM06",
      "Select Revenue Measures": "HSWM05",
      "Social Security": "HSWM01",
      "Trade": "HSWM04",
      "Defense Intelligence and Warfighter Support": "HSIG06",
      "Oversight and Investigations": "HSIG09",
      "Counterterrorism, Counterintelligence, and Counterproliferation": "HSIG05",
      "Intelligence Modernization and Readiness": "HSIG08",
      "Department of Defense Intelligence and Overhead Architecture": "HSIG04",
      "Emerging Threats": "HSIG03",
      "National Security Agency and Cybersecurity": "HSIG02",
      "Central Intelligence Agency": "HSIG01",
      "Strategic Technologies and Advanced Research": "HSIG10",
    };
    if (housesub.hasOwnProperty(t)) {
      console.log("house subcommittee id = "+housesub[t]);
      var y = housesub[t].substring(0,4);
      console.log(y);
      $.ajax({
        url: "https://api.propublica.org/congress/v1/116/house/committees/"+y+"/subcommittees/"+housesub[t]+".json" ,
        type: "GET",
        datatype: "json",
        headers: {'X-API-Key': 'vNgtOv93Sq3FsN6A9xTzD2Ffca8jtk8v7JSneL71'},
      }).done(function (data){
        console.log(data);
        $('#R').append(
          '<h5>'+data.results[0].name+' - Republican'+'</h5>'
        );
        $('#D').append(
          '<h5>'+data.results[0].name+' - Democratic'+'</h5'
        )
        for (r=0; r<data.results[0].current_members.length; r++){
          var name=data.results[0].current_members[r].name;
          var name4link=name.split(' ').slice(-1).join(' ');
          var replastname=name4link.toLowerCase();
          console.log(data.results[0].current_members[r].name);
          if (data.results[0].current_members[r].party === "R") {
            console.log("Found Republican");
            $('#R').append(
              '<a href="https://'+replastname+'.house.gov" target="blank">'+'<img src="/pics/congress_pics/'+data.results[0].current_members[r].id+'.jpg" class="img-fluid rounded mx-auto d-block">'+'</a>'+'<h6>'+data.results[0].current_members[r].name+'</h6>'+      
          '<button style= "background-color:  #006600" type="button" class="btn btn-light" onclick = "inf(\''+data.results[0].current_members[r].id+'\')"> <h5 style="-webkit-text-fill-color: #ffff">INFLUENCE</h5> </button>'
    
            )
          }
          if (data.results[0].current_members[r].party === "D") {
            var name = data.results[0].current_members[r].name;
          var name4link = name.split(' ').slice(-1).join(' ');
            console.log("Found Democrat");
            $('#D').append(
              '<a href="https://'+replastname+'.house.gov" target="blank">'+'<img src="/pics/congress_pics/'+data.results[0].current_members[r].id+'.jpg" class="img-fluid rounded mx-auto d-block">'+'</a>'+'<h6>'+data.results[0].current_members[r].name+'</h6>'+
          '<button style= "background-color:  #006600" type="button" class="btn btn-light" onclick = "inf(\''+data.results[0].current_members[r].id+'\')"> <h5 style="-webkit-text-fill-color: #ffff">INFLUENCE</h5> </button>'
            );
          }
          if (data.results[0].current_members[r].party === "I"){
            var name = data.results[0].current_members[r].name;
          var name4link = name.split(' ').slice(-1).join(' ');
            console.log("Found Independent");
            var random_number = Math.random();
            console.log(random_number);
            if ( random_number < .49 ) {
              $('#R').append(
                '<a href="https://'+replastname+'.house.gov" target="blank">'+'<img src="/pics/congress_pics/'+data.results[0].current_members[r].id+'.jpg" class="img-fluid rounded mx-auto d-block">'+'</a>'+'<h6>'+data.results[0].current_members[r].name+'</h6>'+
          '<button style= "background-color:  #006600" type="button" class="btn btn-light" onclick = "inf(\''+data.results[0].current_members[r].id+'\')"> <h5 style="-webkit-text-fill-color: #ffff">INFLUENCE</h5> </button>'
              )
            } else {
              $('#D').append(
                '<a href="https://'+replastname+'.house.gov" target="blank">'+'<img src="/pics/congress_pics/'+data.results[0].current_members[r].id+'.jpg" class="img-fluid rounded mx-auto d-block">'+'</a>'+'<h6>'+data.results[0].current_members[r].name+'</h6>'+
          '<button style= "background-color:  #006600" type="button" class="btn btn-light" onclick = "inf(\''+data.results[0].current_members[r].id+'\')"> <h5 style="-webkit-text-fill-color: #ffff">INFLUENCE</h5> </button>'
              )
            }
          }
    
        };
    
      })
      $('#R').fadeIn(3000).css('display','block')
        $('#D').fadeIn(3000).css('display','block')
        $('#repcontent').css('display', 'none')
        $('#officials').css('display', 'none')
        $('#about').css('display', 'none')
        $('#home').css('display', 'none')
        $('#lobby').css('display', 'none')
        $('#getlobbydata').css('display', 'none')
        $('#carousel').css('display', 'none')
        $('#caroucontent').css('display', 'none')
        $('#cards').css('display', 'none')
        $('#plan').css('display', 'none')
        $('#sen1').css('display', 'none')
        $('#sen1').empty()
        $('#sen1data').css('display', 'none')
        $('#sen1committeedatabutton').empty()
        $('#sen1subcommitteedatabutton').empty()
        $('#sen2').css('display', 'none')
        $('#sen2').empty()
        $('#sen2data').css('display', 'none')
        $('#sen2committeedatabutton').empty()
        $('#sen2subcommitteedatabutton').empty()
        $('#rep').css('display', 'none')
        $('#rep').empty()
        $('#repcommitteedatabutton').empty()
        $('#repsubcommitteedatabutton').empty()
        $('#repdata').css('display', 'none')
  
    }
    if (housecommittees.hasOwnProperty(f)) {
      window.localStorage.setItem('housecommittee', housecommittees[f]);
      console.log("house committee identified : "+housecommittees[f]);
      $.ajax({
        url: "https://api.propublica.org/congress/v1/116/house/committees/"+window.localStorage.getItem('housecommittee')+".json" ,
        type: "GET",
        datatype: "json",
        headers: {'X-API-Key': 'vNgtOv93Sq3FsN6A9xTzD2Ffca8jtk8v7JSneL71'},
      }).done(function(data) {
        console.log(data);
        $('#R').append( 
          '<h5>'+data.results[0].name+' - Republican'+'</h5>'
        );
        $('#D').append( 
          '<h5>'+data.results[0].name+' - Democratic'+'</h5>'
        )
        for (r=0; r<data.results[0].current_members.length; r++){
          console.log(data.results[0].current_members[r].name);
          if (data.results[0].current_members[r].party === "R") {
            var name = data.results[0].current_members[r].name;
            var name4link = name.split(' ').slice(-1).join(' ');
            var replastname=name4link.toLowerCase();
            console.log("Found Republican");
    
            $('#R').append(
              '<a href="https://'+replastname+'.house.gov" target="blank">'+'<img src="/pics/congress_pics/'+data.results[0].current_members[r].id+'.jpg" class="img-fluid rounded mx-auto d-block">'+'</a>'+'<h6>'+data.results[0].current_members[r].name+'</h6>'+
          '<button style= "background-color:  #006600" type="button" class="btn btn-light" onclick = "inf(\''+data.results[0].current_members[r].id+'\')"> <h5 style="-webkit-text-fill-color: #ffff">INFLUENCE</h5> </button>'
            )
          }
          if (data.results[0].current_members[r].party === "D") {
            var name = data.results[0].current_members[r].name;
            var name4link = name.split(' ').slice(-1).join(' ');
            var replastname=name4link.toLowerCase();
            console.log("Found Democrat");
            $('#D').append(
              '<a href="https://'+replastname+'.house.gov" target="blank">'+'<img src="/pics/congress_pics/'+data.results[0].current_members[r].id+'.jpg" class="img-fluid rounded mx-auto d-block">'+'</a>'+'<h6>'+data.results[0].current_members[r].name+'</h6>'+
          '<button style= "background-color:  #006600" type="button" class="btn btn-light" onclick = "inf(\''+data.results[0].current_members[r].id+'\')"> <h5 style="-webkit-text-fill-color: #ffff">INFLUENCE</h5> </button>'
            )
          }
          if (data.results[0].current_members[r].party === "I"){
            var name = data.results[0].current_members[r].name;
            var name4link = name.split(' ').slice(-1).join(' ');
            var replastname=name4link.toLowerCase();
            console.log("Found Independent");
            var random_number = Math.random();
            console.log(random_number);
            if ( random_number < .49 ) {
              $('#R').append(
                '<a href="https://'+replastname+'.house.gov" target="blank">'+'<img src="/pics/congress_pics/'+data.results[0].current_members[r].id+'.jpg" class="img-fluid rounded mx-auto d-block">'+'</a>'+'<h6>'+data.results[0].current_members[r].name+'</h6>'+
          '<button style= "background-color:  #006600" type="button" class="btn btn-light" onclick = "inf(\''+data.results[0].current_members[r].id+'\')"> <h5 style="-webkit-text-fill-color: #ffff">INFLUENCE</h5> </button>'
              )
            } else {
              $('#D').append(
                '<a href="https://'+replastname+'.house.gov" target="blank">'+'<img src="/pics/congress_pics/'+data.results[0].current_members[r].id+'.jpg" class="img-fluid rounded mx-auto d-block">'+'</a>'+'<h6>'+data.results[0].current_members[r].name+'</h6>'+
          '<button style= "background-color:  #006600" type="button" class="btn btn-light" onclick = "inf(\''+data.results[0].current_members[r].id+'\')"> <h5 style="-webkit-text-fill-color: #ffff">INFLUENCE</h5> </button>'
              )
            }
          }
    
        };
      })
        $('#repcontent').css('display', 'none')
        $('#R').fadeIn(3000).css('display','block'),
        $('#D').fadeIn(3000).css('display','block'),
        $('#officials').css('display', 'none')
        $('#about').css('display', 'none')
        $('#home').css('display', 'none')
        $('#lobby').css('display', 'none')
        $('#getlobbydata').css('display', 'none')
        $('#carousel').css('display', 'none')
        $('#caroucontent').css('display', 'none')
        $('#cards').css('display', 'none')
        $('#plan').css('display', 'none')
        $('#sen1').css('display', 'none')
        $('#sen1').empty()
        $('#sen1data').css('display', 'none')
        $('#sen1committeedatabutton').empty()
        $('#sen1subcommitteedatabutton').empty()
        $('#sen2').css('display', 'none')
        $('#sen2').empty()
        $('#sen2data').css('display', 'none')
        $('#sen2committeedatabutton').empty()
        $('#sen2subcommitteedatabutton').empty()
        $('#rep').css('display', 'none')
        $('#rep').empty()
        $('#repcommitteedatabutton').empty()
        $('#repsubcommitteedatabutton').empty()
        $('#repdata').css('display', 'none')
        $('#sen1table').empty()
        $('#sen2table').empty()
        $('#reptable').empty()
        $('#inf').css('display', 'none')
    
    } else {
      console.log("Where'd you mess up")
    }
  
  
  }
  //Converting for the ID of and Finding Committee & Subcommittee 
  function oo(x, g) {
  var senatecommittees = {
    "Committee on Agriculture, Nutrition, and Forestry":"SSAF",
    "Committee on Appropriations": "SSAP",
    "Committee on Armed Services": "SSAS",
    "Committee on Banking, Housing, and Urban Affairs": "SSBK",
    "Committee on Commerce, Science, and Transportation": "SSCM",
    "Committee on Energy and Natural Resources": "SSEG",
    "Committee on Environment and Public Works": "SSEV",
    "Committee on Finance": "SSFI",
    "Committee on Foreign Relations": "SSFR",
    "Committee on Health, Education, Labor, and Pensions": "SSHR",
    "Committee on Homeland Security and Governmental Affairs": "SSGA",
    "Committee on Indian Affairs": "SLIA",
    "Committee on Rules and Administration": "SSRA",
    "Committee on Small Business and Entrepreneurship": "SSSB",
    "Committee on the Budget":  "SSBU",
    "Committee on the Judiciary": "SSJU",
    "Committee on Veterans Affairs": "SSVA",
    "Select Committee on Ethics": "SLET",
    "Select Committee on Intelligence": "SLIN",
    "Special Committee on Aging": "SPAG",
    "United States Caucus on International Narcotics Control": "SCNC",
  }
  var senatesub = {
    "Commodities, Risk Management, and Trade": "SSAF13",
    "Conservation, Forestry, and Natural Resources": "SSAF14",
    "Rural Development and Energy": "SSAF15",
    "Livestock, Marketing, and Agriculture Security": "SSAF17",
    "Nutrition, Agricultural Research, and Specialty Crops": "SSAF16",
    "Departments of Labor, Health and Human Services, and Education, and Related Agencies": "SSAP18",
    "Commerce, Justice, Science, and Related Agencies Subcommittee": "SSAP16",
    "Energy and Water Development Subcommittee": "SSAP22",
    "State, Foreign Operations, and Related Programs": "SSAP20",
    "Department of Homeland Security Subcommittee": "SSAP14",
    "Department of the Interior, Environment, and Related Agencies": "SSAP17",
    "Transportation, Housing and Urban Development, and Related Agencies": "SSAP24",
    "Legislative Branch Subcommittee": "SSAP08",
    "Agriculture, Rural Development, Food and Drug Administration, and Related Agencies": "SSAP01",
    "Department of Defense Subcommittee": "SSAP02",
    "Military Construction and Veterans Affairs, and Related Agencies": "SSAP19",
    "Financial Services and General Government": "SSAP23",
    "Strategic Forces Subcommittee": "SSAS16",
    "Emerging Threats and Capabilities": "SSAS20",
    "SeaPower Subcommittee": "SSAS13",
    "Airland Subcommittee": "SSAS14",
    "Readiness and Management Support Subcommittee": "SSAS15",
    "Personnel Subcommittee": "SSAS17",
    "Cybersecurity": "SSAS21",
    "Securities, Insurance, and Investment": "SSBK04",
    "Housing, Transportation, and Community Development": "SSBK09",
    "Economic Policy": "SSBK12",
    "National Security and International Trade and Finance": "SSBK05",
    "Financial Institutions and Consumer Protection": "SSBK08",
    "Communications, Technology, Innovation, and the Internet": "SSCM26",
    "Transportation and Safety Subcommittee": "SSCM32",
    "Security Subcommittee": "SSCM31",
    "Manufacturing, Trade, and Consumer Protection Subcommittee": "SSCM29",
    "Aviation and Space Subcommittee": "SSCM28",
    "Science, Oceans, Fisheries, and Weather": "SSCM30",
    "Public Lands, Forests, and Mining Subcommittee": "SSEG03",
    "National Parks": "SSEG04",
    "Water and Power Subcommittee": "SSEG07",
    "Energy Subcommittee": "SSEG01",
    "Fisheries, Water, and Wildlife": "SSEV15",
    "Superfund, Waste Management, and Regulatory Oversight": "SSEV09",
    "Clean Air and Nuclear Safety": "SSEV10",
    "Green Jobs and the New Economy": "SSEV16",
    "Transportation and Infrastructure": "SSEV08",
    "Oversight": "SSEV18",
    "International Trade, Customs, and Global Competitiveness": "SSFI13",
    "Energy, Natural Resources, and Infrastructure": "SSFI12",
    "Fiscal Responsibility and Economic Growth": "SSFI14",
    "Health Care": "SSFI10",
    "Social Security, Pensions, and Family Policy": "SSFI02",
    "Taxation and IRS Oversight": "SSFI11",
    "International Development and Foreign Assistance, Economic Affairs, International Environmental Protection, and Peace Corps": "SSFR12",
    "International Operations and Organizations, Human Rights, Democracy, and Global Womens Issues": "SSFR13",
    "Africa and Global Health Policy: SSFR09 Europe and Regional Security Cooperation": "SSFR01",
    "East Asia, the Pacific, and International Cybersecurity Policy": "SSFR02",
    "Near East, South Asia, Central Asia, and Counterterrorism": "SSFR07",
    "Western Hemisphere, Transnational Crime, Civilian Security, Democracy, Human Rights, and Global Womens Issues": "SSFR06",
    "State Department and USAID Management, International Operations, and Bilateral International Development": "SSFR14",
    "Multilateral International Development, Multilateral Institutions, and International Economic, Energy, and Environmental Policy": "SSFR15",
    "Children and Families: SSHR09 Employment and Workplace Safety": "SSHR11",
    "Primary Health and Retirement Security": "SSHR12",
    "Permanent Subcommittee on Investigations": "SSGA01",
    "Financial and Contracting Oversight": "SSGA15",
    "Emergency Management, Intergovernmental Relations, and the District of Columbia": "SSGA17",
    "the Efficiency and Effectiveness of Federal Programs and the Federal Workforce": "SSGA16",
    "Regulatory Affairs and Federal Management Subcommittee": "SSGA19",
    "Federal Spending Oversight and Emergency Management": "SSGA18",
  };
  if (senatecommittees.hasOwnProperty(x)) {
    window.localStorage.setItem('senatecommittee', senatecommittees[x]);
    console.log("senate committee identified : " + senatecommittees[x]);
    $.ajax({
      url: "https://api.propublica.org/congress/v1/116/senate/committees/"+window.localStorage.getItem('senatecommittee')+".json" ,
      type: "GET",
      datatype: "json",
      headers: {'X-API-Key': 'vNgtOv93Sq3FsN6A9xTzD2Ffca8jtk8v7JSneL71'},
    }).done(function(data) {
      console.log(data);
      $('#R').append( 
        '<h5>'+data.results[0].name+' - (R)'+'</h5>'
      );
      $('#D').append( 
        '<h5>'+data.results[0].name+' - (D)'+'</h5>'
      )
      
  
      for (r=0; r<data.results[0].current_members.length; r++){
        console.log(data.results[0].current_members[r].name);
        if (data.results[0].current_members[r].party === "R") {
          var name = data.results[0].current_members[r].name;
          var name4link = name.split(' ').slice(-1).join(' ');
          console.log("Found Republican");
          $('#R').append(
            '<a href="https://www.'+name4link+'.senate.gov" target="blank">'+'<img src="/pics/congress_pics/'+data.results[0].current_members[r].id+'.jpg" class="img-fluid rounded mx-auto d-block">'+'</a>'+'<h6>'+data.results[0].current_members[r].name+'</h6>'+      
        '<button style= "background-color:  #006600" type="button" class="btn btn-light" onclick = "inf(\''+data.results[0].current_members[r].id+'\')"> <h5 style="-webkit-text-fill-color: #ffff">INFLUENCE</h5> </button>'
  
          )
        }
        if (data.results[0].current_members[r].party === "D") {
          var name = data.results[0].current_members[r].name;
          var name4link = name.split(' ').slice(-1).join(' ');
          console.log("Found Democrat");
          $('#D').append(
            '<a href="https://www.'+name4link+'.senate.gov" target="blank">'+'<img src="/pics/congress_pics/'+data.results[0].current_members[r].id+'.jpg" class="img-fluid rounded mx-auto d-block">'+'</a>'+'<h6>'+data.results[0].current_members[r].name+'</h6>'+
        '<button style= "background-color:  #006600" type="button" class="btn btn-light" onclick = "inf(\''+data.results[0].current_members[r].id+'\')"> <h5 style="-webkit-text-fill-color: #ffff">INFLUENCE</h5> </button>'
          );
        }
        if (data.results[0].current_members[r].party === "I"){
          console.log("Found Independent");
          var name = data.results[0].current_members[r].name;
          var name4link = name.split(' ').slice(-1).join(' ');
          var random_number = Math.random();
          console.log(random_number);
          if ( random_number < .49 ) {
            $('#R').append(
              '<a href="https://www.'+name4link+'.senate.gov" target="blank">'+'<img src="/pics/congress_pics/'+data.results[0].current_members[r].id+'.jpg" class="img-fluid rounded mx-auto d-block">'+'</a>'+'<h6>'+data.results[0].current_members[r].name+'</h6>'+
        '<button style= "background-color:  #006600" type="button" class="btn btn-light" onclick = "inf(\''+data.results[0].current_members[r].id+'\')"> <h5 style="-webkit-text-fill-color: #ffff">INFLUENCE</h5> </button>'
            )
          } else {
            $('#D').append(
              '<a href="https://www.'+name4link+'.senate.gov" target="blank">'+'<img src="/pics/congress_pics/'+data.results[0].current_members[r].id+'.jpg" class="img-fluid rounded mx-auto d-block">'+'</a>'+'<h6>'+data.results[0].current_members[r].name+'</h6>'+
        '<button style= "background-color:  #006600" type="button" class="btn btn-light" onclick = "inf(\''+data.results[0].current_members[r].id+'\')"> <h5 style="-webkit-text-fill-color: #ffff">INFLUENCE</h5> </button>'
            )
          }
        }
  
      };
    })
    $('#R').fadeIn(3000).css('display','block'),
      $('#D').fadeIn(3000).css('display','block'),
      $('#officials').css('display', 'none')
      $('#about').css('display', 'none')
      $('#home').css('display', 'none')
      $('#lobby').css('display', 'none')
      $('#getlobbydata').css('display', 'none')
      $('#carousel').css('display', 'none')
      $('#caroucontent').css('display', 'none')
      $('#cards').css('display', 'none')
      $('#plan').css('display', 'none')
      $('#sen1').css('display', 'none')
      $('#sen1').empty()
      $('#sen1data').css('display', 'none')
      $('#sen1committeedatabutton').empty()
      $('#sen1subcommitteedatabutton').empty()
      $('#sen2').css('display', 'none')
      $('#sen2').empty()
      $('#sen2data').css('display', 'none')
      $('#sen2committeedatabutton').empty()
      $('#sen2subcommitteedatabutton').empty()
      $('#rep').css('display', 'none')
      $('#rep').empty()
      $('#repcommitteedatabutton').empty()
      $('#repsubcommitteedatabutton').empty()
      $('#repdata').css('display', 'none')
  } else {
  }
  if (senatesub.hasOwnProperty(g)) {
    console.log("house subcommittee id = "+senatesub[g]);
    var y = senatesub[g].substring(0,4);
    console.log(y);
    $.ajax({
      url: "https://api.propublica.org/congress/v1/116/house/committees/"+y+"/subcommittees/"+senatesub[g]+".json" ,
      type: "GET",
      datatype: "json",
      headers: {'X-API-Key': 'vNgtOv93Sq3FsN6A9xTzD2Ffca8jtk8v7JSneL71'},
    }).done(function (data){
      console.log(data);
      $('#R').append(
        '<h5>'+data.results[0].name+' - (R)'+'</h5>'
      );
      $('#D').append(
        '<h5>'+data.results[0].name+' - (D)'+'</h5'
      )
      for (r=0; r<data.results[0].current_members.length; r++){
        console.log(data.results[0].current_members[r].name);
        if (data.results[0].current_members[r].party === "R") {
        var name = data.results[0].current_members[r].name;
        var name4link = name.split(' ').slice(-1).join(' ');
          console.log("Found Republican");
          $('#R').append(
            '<a href="https://www.'+name4link+'.senate.gov" target="blank">'+'<img src="/pics/congress_pics/'+data.results[0].current_members[r].id+'.jpg" class="img-fluid rounded mx-auto d-block">'+'</a>'+'<h6>'+data.results[0].current_members[r].name+'</h6>'+      
        '<button style= "background-color:  #006600" type="button" class="btn btn-light" onclick = "inf(\''+data.results[0].current_members[r].id+'\')"> <h5 style="-webkit-text-fill-color: #ffff">INFLUENCE</h5> </button>'
  
          )
        }
        if (data.results[0].current_members[r].party === "D") {
        var name = data.results[0].current_members[r].name;
        var name4link = name.split(' ').slice(-1).join(' ');
          console.log("Found Democrat");
          $('#D').append(
            '<a href="https://www.'+name4link+'.senate.gov" target="blank">'+'<img src="/pics/congress_pics/'+data.results[0].current_members[r].id+'.jpg" class="img-fluid rounded mx-auto d-block">'+'</a>'+'<h6>'+data.results[0].current_members[r].name+'</h6>'+
        '<button style= "background-color:  #006600" type="button" class="btn btn-light" onclick = "inf(\''+data.results[0].current_members[r].id+'\')"> <h5 style="-webkit-text-fill-color: #ffff">INFLUENCE</h5> </button>'
          );
        }
        if (data.results[0].current_members[r].party === "I"){
          console.log("Found Independent");
          var random_number = Math.random();
          console.log(random_number);
          if ( random_number < .49 ) {
            $('#R').append(
              '<a href="https://www.'+name4link+'.senate.gov" target="blank">'+'<img src="/pics/congress_pics/'+data.results[0].current_members[r].id+'.jpg" class="img-fluid rounded mx-auto d-block">'+'</a>'+'<h6>'+data.results[0].current_members[r].name+'</h6>'+
        '<button style= "background-color:  #006600" type="button" class="btn btn-light" onclick = "inf(\''+data.results[0].current_members[r].id+'\')"> <h5 style="-webkit-text-fill-color: #ffff">INFLUENCE</h5> </button>'
            )
          } else {
            $('#D').append(
              '<a href="https://www.'+name4link+'.senate.gov" target="blank">'+'<img src="/pics/congress_pics/'+data.results[0].current_members[r].id+'.jpg" class="img-fluid rounded mx-auto d-block">'+'</a>'+'<h6>'+data.results[0].current_members[r].name+'</h6>'+
        '<button style= "background-color:  #006600" type="button" class="btn btn-light" onclick = "inf(\''+data.results[0].current_members[r].id+'\')"> <h5 style="-webkit-text-fill-color: #ffff">INFLUENCE</h5> </button>'
            )
          }
        }
  
      };
  
    })
    $('#R').fadeIn(3000).css('display','block'),
      $('#D').fadeIn(3000).css('display','block'),
      $('#officials').css('display', 'none')
      $('#about').css('display', 'none')
      $('#home').css('display', 'none')
      $('#lobby').css('display', 'none')
      $('#getlobbydata').css('display', 'none')
      $('#carousel').css('display', 'none')
      $('#caroucontent').css('display', 'none')
      $('#cards').css('display', 'none')
      $('#plan').css('display', 'none')
      $('#sen1').css('display', 'none')
      $('#sen1').empty()
      $('#sen1data').css('display', 'none')
      $('#sen1committeedatabutton').empty()
      $('#sen1subcommitteedatabutton').empty()
      $('#sen2').css('display', 'none')
      $('#sen2').empty()
      $('#sen2data').css('display', 'none')
      $('#sen2committeedatabutton').empty()
      $('#sen2subcommitteedatabutton').empty()
      $('#rep').css('display', 'none')
      $('#rep').empty()
      $('#repcommitteedatabutton').empty()
      $('#repsubcommitteedatabutton').empty()
      $('#repdata').css('display', 'none')
  };
  $('#repcontent').css('display', 'none')
  };
  

  function initMap() {
    map = new google.maps.Map(document.getElementById('map'), {
      center: {lat: 38.907, lng: -77.036},
      zoom: 10
    });
    var infowindow = new google.maps.InfoWindow;
    var geocoder = new google.maps.Geocoder;
    
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(function(position) {
        var pos = {
          lat: position.coords.latitude,
          lng: position.coords.longitude,
        };
        console.log(position.coords.latitude + ", " + position.coords.longitude); 
        var latlng = {lat: pos.lat, lng: pos.lng};
        geocoder.geocode({'location': latlng}, function(results, status) {
      if (status === 'OK') {
        if (results[0]) {
          map.setZoom(11);
          var marker = new google.maps.Marker({
            position: latlng,
            map: map
          });
          globalStringAddress=results[0].formatted_address,
          console.log("address is : "+ globalStringAddress);
          infowindow.setContent(results[0].formatted_address);
          infowindow.open(map, marker);
          var globalStringAddress = globalStringAddress.replace(/[\s]/g, '+').replace(/[,]/g, '%2C');
          // window.localStorage.setItem('globaladdress', globalStringAddress);
          var civicAPI ="https://www.googleapis.com/civicinfo/v2/representatives?address="+globalStringAddress+"&key=AIzaSyBwA2-va1J2oaO3IhPn2xqItnyUyhkfkqk";
          console.log(civicAPI);
          $.getJSON(civicAPI, function(result){
            window.localStorage.setItem('Senator1', result.officials[2].name),
            window.localStorage.setItem('Senator2', result.officials[3].name),
            window.localStorage.setItem('Rep', result.officials[4].name),
            Senator1= result.officials[2].name,
            Senator2= result.officials[3].name,
            Rep= result.officials[4].name
});





        } else {
          window.alert('No results found');
        }
      } else {
        window.alert('Geocoder failed due to: ' + status);
      }
    });
    

      
        infowindow.setPosition(pos);
        infowindow.setContent('Cool House Loser');
        infowindow.open(map);
        map.setCenter(pos);
      }, function() {
        handleLocationError(true, infowindow, map.getCenter());
      });
    } else {
      // Browser doesn't support Geolocation
      handleLocationError(false, infowindow, map.getCenter());
    }
  }
  function handleLocationError(browserHasGeolocation, infoWindow, pos) {
    infoWindow.setPosition(pos);
    infoWindow.setContent(browserHasGeolocation ?
                          'Error: The Geolocation service failed.' :
                          'Error: Your browser doesn\'t support geolocation.');
    infoWindow.open(map);
      }
  
  //ProPublica API for Rep and Senator DaTaBeGiN
  $(document).ready(function(){
    $('#getdata').click(function() {
      $('#officials').css('display', 'none'),
      $('#about').css('display', 'none')  
      $('#home').css('display', 'none')
      $('#lobby').css('display', 'none')
      $('#contact').css('display', 'none')
      $('#carousel').css('display', 'none')
      $('#caroucontent').css('display', 'none')
      $('#cards').css('display', 'none')
      $('#plan').css('display', 'none')
      $('#getlobbydata').css('display', 'none')
      $('#sen1').empty()
      $('#sen1committeedatabutton').empty()
      $('#sen1subcommitteedatabutton').empty()
      $('#sen2').empty()
      $('#sen2committeedatabutton').empty()
      $('#sen2subcommitteedatabutton').empty()
      $('#rep').empty()
      $('#repcommitteedatabutton').empty()
      $('#repsubcommitteedatabutton').empty()
      $('#D').css('display', 'none')
      $('#D').empty()
      $('#R').css('display', 'none')
      $('#R').empty()
      $('#inf').css('display', 'none')
      $('#sen1table').empty()
      $('#sen2table').empty()
      $('#reptable').empty()
      $('#inf').empty()
      $('#repcontent').css('display', 'block')
      $('#load').css('display', 'block')
      ff();
  
        //IF ERROR
      if (window.localStorage.getItem('S1ID')==='fail') {
                    $("#lobby").css(
                      'display', 'none'
                      )
                      window.localStorage.setItem('RepIDs', 'Failure');
                      q= "WHAT DID YOU DO";
                      headerphoto1='Sweet Fancy Moses';
                      captionphoto1='Local person breaks coolest thing to happen to the internet since people realized you could point cameras at cats. (Also yes good boy dog woof bark yes) ';
                      linkphoto1='https://media.giphy.com/media/Pk20jMIe44bHa/giphy.gif';
                      headerphoto2='Shame brought to local persons family';
                      captionphoto2='Shame today ladies and gents. Pure shame was brought down upon a local family after someone broke the DS all but ensuring the apocalypse';
                      linkphoto2='https://media.giphy.com/media/jEkFtJMJuNTgc/giphy.gif';
                      headerphoto3='Christmas cancelled after reckless actions of local internet consumer';
                      captionphoto3='Santa just called us and said your actions are reprehensable. With Christmas out the window he will now consult with the Easter bunny about a possible withdrawal';
                      linkphoto3='https://heavyeditorial.files.wordpress.com/2018/12/GettyImages-1074396864.jpg?quality=65&strip=all';
                      var cstrap = {
              code:  '<div class="column">' +
            '<div class="col-lg-12 text-center">' +
              '<h1 class="mt-5">'+q+'</h1>'+
              '<div id="carouselExampleIndicators" class="carousel slide" data-ride="carousel">' +
                      '<ol class="carousel-indicators">'+
                        '<li data-target="#carouselExampleIndicators" data-slide-to="0" class="active"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="1"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="2"></li>' +
                      '</ol>' +
                      '<div class="carousel-inner">' +
                        '<div class="carousel-item active">' +
                              '<div class="carousel-caption d-none d-md-block">' +
                                      '<h5>'+headerphoto1+'</h5>' +
                                      '<p1>'+captionphoto1+'</p1>' +
                              '</div>' +
                          '<img class="d-block w-100" src="'+linkphoto1+'" alt="First slide">' +
                        '</div>' +
                        '<div class="carousel-item">' +
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto2+'</h5>'+
                                      '<p1>'+captionphoto2+'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto2+'" alt="Second slide">'+
                        '</div>'+
                        '<div class="carousel-item">'+
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto3+'</h5>'+
                                      '<p1>'+captionphoto3 +'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto3+'" alt="Third slide">' +
                        '</div>'+
                      '</div>'+
                      '<a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">'+
                        '<span class="carousel-control-prev-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Previous</span>'+
                      '</a>'+
                      '<a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">'+
                        '<span class="carousel-control-next-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Next</span>'+
                      '</a>'+
              '</div>'
              };
                      var u1 = cstrap.code;
                      document.getElementById("carousel").innerHTML=u1;
                      $('#carousel').fadeIn(2000);
                      var u = {
                      rhetoric: `ERROAR BING BONG BOP. Looks like a fumble on our end. Try back later.`,
                      sarcasm: "Also, we know what the error was now. It was your fault. Nice job. You've ruined it for everyone"
                      };
                      document.getElementById("content").innerHTML=u.rhetoric;
                      $('#content').fadeIn(2000);
                      $('#repcontent').css('display', 'none')
                      $('#load').css('display', 'none')
      } else { 
      $.ajax({
        url: "https://api.propublica.org/congress/v1/members/"+window.localStorage.getItem('S1ID')+".json", 
        type: "GET",
        dataType: 'json',
        headers: {'X-API-Key': 'vNgtOv93Sq3FsN6A9xTzD2Ffca8jtk8v7JSneL71'}
      }).done(function(data){
        window.localStorage.setItem('sen1party', data.results[0].current_party)
        console.log(data);
        $('#sen1').delay(2000).fadeIn(2000),                                                          
        $('#sen1').append(
          
          '<a href="https://www.'+data.results[0].last_name+'.senate.gov" target="blank">'+'<img src="/pics/congress_pics/'+window.localStorage.getItem('S1ID')+'.jpg" class="img-fluid rounded mx-auto d-block">'+'</a>'+
          '<div class= "col text-center">'+'<h6>'+data.results[0].first_name+" "+data.results[0].last_name+'</h6>'+
          '<button type="button" class="btn"  onclick= "inf(\''+data.results[0].member_id+'\')"> <h5>INFLUENCE</h5> </button>'+
          '</div>'
  
        );
        function jj(x){
            for (i=0; i<data.results[0].roles[0].committees.length; i++){
              console.log(data.results[0].roles[0].committees[i].name);
              var committee = data.results[0].roles[0].committees[i].name;
              if (committee.indexOf("'")>-1){
                  var cleaned = committee.replace("'", "")
                  $('#sen1committeedatabutton').append(
                    `<a class="dropdown-item"  onclick="oo(`+"'"+cleaned+"'"+`)">`+committee+'</a>',
                  ) 
                }
              else{
                $('#sen1committeedatabutton').append(
                    `<a class="dropdown-item"  onclick="oo(`+"'"+committee+"'"+`)">`+committee+'</a>',
                  ) 
              }

            };
            for (j=0; j<data.results[0].roles[0].subcommittees.length; j++) {
              console.log(data.results[0].roles[0].subcommittees[j].name);
              var subcommittee = data.results[0].roles[0].subcommittees[j].name
              if (subcommittee.indexOf("'">-1)){
                var cleaned = subcommittee.replace("'", "")
                $('#sen1subcommitteedatabutton').append(
                    '<a class="dropdown-item" onclick="oo('+"null, "+"'"+cleaned+"'"+')">'+subcommittee+'</a>',

                )
              } else{
                $('#sen1subcommitteedatabutton').append(
                  '<a class="dropdown-item" onclick="oo('+"null, "+"'"+subcommittee+"'"+')">'+subcommittee+'</a>', 
                )
              };
              
            }
        };
        jj();
         $('#sen1data').delay(2000).fadeIn(2000);
         $('#sen1table').prepend(
         );
         if (data.results[0].roles[0].votes_against_party_pct > 15.00) {
           $('#sen1table').delay(10000).prepend(
            '<div class="alert alert-info" role="alert" onclick= "inf(\''+data.results[0].member_id+'\')">'+ "NERD ALERT ~"
    +data.results[0].first_name+" "+data.results[0].last_name+
  '</div>'
           )
         }
       })
      }
      $.ajax({
        url: "https://api.propublica.org/congress/v1/members/"+window.localStorage.getItem('S2ID')+".json", 
        type: "GET",
        dataType: 'json',
        headers: {'X-API-Key': 'vNgtOv93Sq3FsN6A9xTzD2Ffca8jtk8v7JSneL71'}
     }).done(function(data){
        window.localStorage.setItem('sen2party', data.results[0].current_party);
       console.log(data);
       $('#sen2').delay(2000).fadeIn(2000),
       $('#sen2').append(
        '<div class= "col text-center">'
        +'<a href="https://www.'+data.results[0].last_name+'.senate.gov" target="blank">'+'<img src="/pics/congress_pics/'+window.localStorage.getItem('S2ID')+'.jpg" class="img-fluid rounded mx-auto d-block">'+'</a>'+
        `<h6>`+data.results[0].first_name+" "+data.results[0].last_name+`</h6>`+
        '<button type="button" class="btn" onclick= "inf(\''+data.results[0].member_id+'\')"> <h5>INFLUENCE</h5> </button>'+
        '</div>'
  
  
       );
       function jj(x){
        for (i=0; i<data.results[0].roles[0].committees.length; i++){
          console.log(data.results[0].roles[0].committees[i].name);
          var committee = data.results[0].roles[0].committees[i].name;
          if (committee.indexOf("'")>-1){
              var cleaned = committee.replace("'", "")
              $('#sen2committeedatabutton').append(
                `<a class="dropdown-item"  onclick="oo(`+"'"+cleaned+"'"+`)">`+committee+'</a>',
              ) 
            }
          else{
            $('#sen2committeedatabutton').append(
                `<a class="dropdown-item"  onclick="oo(`+"'"+committee+"'"+`)">`+committee+'</a>',
              ) 
          }

        };
        for (j=0; j<data.results[0].roles[0].subcommittees.length; j++) {
          console.log(data.results[0].roles[0].subcommittees[j].name);
          $('#sen2subcommitteedatabutton').append(
            '<a class="dropdown-item" onclick="oo('+"null, "+"'"+data.results[0].roles[0].subcommittees[j].name+"'"+')">'+data.results[0].roles[0].subcommittees[j].name+'</a>',
          )
        }
    };
        jj();

       $('#sen2data').delay(2000).fadeIn(2000);
       $('#sen2table').prepend(
         );
         if (data.results[0].roles[0].votes_against_party_pct > 15.00) {
           $('#sen2table').delay(5000).prepend(
            '<div class="alert alert-info" role="alert" onclick= "inf(\''+data.results[0].member_id+'\')">'+ "Looks like "
    +data.results[0].first_name+" "+data.results[0].last_name+" could be in play. click here for more info"+
  '</div>'
           )
         }
      $.ajax({
        url: "https://api.propublica.org/congress/v1/members/"+window.localStorage.getItem('RID')+".json", 
        type: "GET",
        dataType: 'json',
        headers: {'X-API-Key': 'vNgtOv93Sq3FsN6A9xTzD2Ffca8jtk8v7JSneL71'}
       }).done(function(data){
        window.localStorage.setItem('repParty', data.results[0].current_party)
         console.log(data);
         var replastname=(data.results[0].last_name).toLowerCase();
         $('#rep').delay(2000).fadeIn(2000),
         $('#rep').append(
          '<div class= "col text-center">'+
          '<a href="https://'+replastname+'.house.gov" target="blank">'+'<img src="/pics/congress_pics/'+window.localStorage.getItem('RID')+'.jpg" class="img-fluid rounded mx-auto d-block">'+'</a>'+
          `<h6>`+data.results[0].first_name+" "+data.results[0].last_name+`</h6>`+
          '<button type="button" class="btn" onclick= "inf(\''+data.results[0].member_id+'\')"> <h5>INFLUENCE</h5> </button>'
            +'</div>'
  
         );
         function jj(x){
            for (i=0; i<data.results[0].roles[0].committees.length; i++){
              console.log(data.results[0].roles[0].committees[i].name);
              var committee = data.results[0].roles[0].committees[i].name;
              if (committee.indexOf("'")>-1){
                  var cleaned = committee.replace("'", "")
                  $('#repcommitteedatabutton').append(
                    `<a class="dropdown-item"  onclick="ooo(`+"'"+cleaned+"'"+`)">`+cleaned+'</a>',
                  ) 
                }
              else{
                $('#repcommitteedatabutton').append(
                    `<a class="dropdown-item"  onclick="ooo(`+"'"+committee+"'"+`)">`+data.results[0].roles[0].committees[i].name+'</a>',
                  ) 
              }

            };
            for (j=0; j<data.results[0].roles[0].subcommittees.length; j++) {
              console.log(data.results[0].roles[0].subcommittees[j].name);
              $('#repsubcommitteedatabutton').append(
                '<a class="dropdown-item" onclick="ooo('+"null, "+"'"+data.results[0].roles[0].subcommittees[j].name+"'"+')">'+data.results[0].roles[0].subcommittees[j].name+'</a>',
              )
            }
        };
        jj(data);
        $('#load').css('display', 'none')
         $('#repdata').delay(2000).fadeIn(2000);
         $('#reptable').prepend(
         );
         if (data.results[0].roles[0].votes_against_party_pct > 15.00 && window.localStorage.getItem('Path') ==! window.localStorage.getItem('repParty') ) {
           $('#reptable').delay(5000).prepend(
            '<div class="alert alert-info" role="alert" onclick= "inf(\''+data.results[0].member_id+'\')">'+ "Looks like "
    +data.results[0].first_name+" "+data.results[0].last_name+" could be in play. click here for more info"+
  '</div>'
           )
         }
       }) 
     })


    })
    $('#load').css('display', 'none')
  })
  
  // INFLUENCE PROFILE 
  var inf = function(x) {
    //x is member_id
  console.log("INF STARTED");
  console.log(x);
var loading = "loading.gif"; 
  $('#sen1').css('display', 'none')
  $('#sen1').empty()
  $('#sen1data').css('display', 'none')
  $('#sen1committeedatabutton').empty()
  $('#sen1subcommitteedatabutton').empty()
  $('#sen2').css('display', 'none')
  $('#sen2').empty()
  $('#sen2data').css('display', 'none')
  $('#sen2committeedatabutton').empty()
  $('#sen2subcommitteedatabutton').empty()
  $('#rep').css('display', 'none')
  $('#rep').empty()
  $('#repcommitteedatabutton').empty()
  $('#repsubcommitteedatabutton').empty()
  $('#repdata').css('display', 'none')
  $('#D').empty()
  $('#D').css('display', 'none')
  $('#R').empty()
  $('#R').css('display', 'none')
  $('#sen1table').empty()
  $('#sen2table').empty()
  $('#reptable').empty()
  $('#firstcontent').css('display', 'none')
  $('#repcontent').css('display', 'none')
  $('#inf').css('display', 'block')
  $('#inf').append(
    '<div id="load" class="row justify-content-center">'+
    '<img src="loading.gif" alt="Loading....">'+
  '</div>'
  )


  console.log('polished off the empties ;)');
  
  
  
  //PROPUBLICA API
  
  $.ajax({
    url: "https://api.propublica.org/congress/v1/members/"+x+".json", 
    type: "GET",
    dataType: 'json',
    headers: {'X-API-Key': 'vNgtOv93Sq3FsN6A9xTzD2Ffca8jtk8v7JSneL71'}
  }).done(function(data) {
    localStorage.setItem('crp_id', data.results[0].crp_id);
    console.log(data);
    

  // OPENSECRETS API's

   $.getJSON( "https://www.opensecrets.org/api/?method=candSummary&cid="+localStorage.getItem("crp_id")+"&cycle=2020&output=json&apikey=dcf447b80edb32e28cb162c2d0548e77").done( function ( opensecretsCand ) {
    $.getJSON( "https://www.opensecrets.org/api/?method=candIndustry&cid="+localStorage.getItem("crp_id")+"&cycle=2020&output=json&apikey=dcf447b80edb32e28cb162c2d0548e77").done( function ( opensecretsInd ) {
      $.getJSON( "https://www.opensecrets.org/api/?method=candContrib&cid="+localStorage.getItem("crp_id")+"&cycle=2020&output=json&apikey=dcf447b80edb32e28cb162c2d0548e77").done( function ( opensecretsContrib ) { 
    
    var Nom = opensecretsCand.response.summary['@attributes'].cand_name;
    var name = Nom.substr(Nom.indexOf(',')+1 ) +" "+ Nom.substr(0, Nom.indexOf(',') );
    var lastpart = Nom.substr(0, Nom.indexOf(',') );
    if( data.results[0].roles[0].chamber === "Senate")    {
      var lastpart = "https://www."+Nom.substr(0, Nom.indexOf(',') )+".senate.gov";
  }
    else
        {
          var lastpart = "https://www."+Nom.substr(0, Nom.indexOf(',') )+".house.gov";
        };

    console.log(opensecretsInd);
    console.log(opensecretsContrib);
    console.log(name);
     
      for (b=0; b<opensecretsInd.response.industries.industry.length; b++) {
          var code = opensecretsInd.response.industries.industry[b]['@attributes'].industry_code;
          console.log(code);
         
                   // THE PART THAT'S BEING A FAG

          // for (q=0; q<industrycodes.length; q++) {
          //   if (q="Catcode") {
          //     console.log('Progress')
          //   } else {
          //     console.log("YOU BLOW")
          //   }
          // };

          function herbert(_code,_Catcode,_json_obj) {
            for (g=0; g<_json_obj.length; g++){
              console.log(g);
              var parent = this.window.parent;
              var ind_code = parent.industrycodes;
              if (_json_obj[g]._Catcode == _code ) {
                  console.log("industry match");
                } 
            };
          };


          // function herbert(x, y) {
          //       for (g=0; g<y.length; g++){
          //         console.log(g);
          //         if (y[g].hasOwnProperty(x)) {
          //             console.log("match");
          //           } else {
          //             console.log("no mas");
          //           }
          //       };
          //     };
          
          //herbert(code,industrycodes);
          herbert(code,"Catcode", industrycodes);
    
    
    
    
    
      };

    $('#inf').empty()    
    $('#inf').append(
    '<div class="row">'+
        '<div style="position: sticky" class="col-md-4">'+
          '<h6>'+name+'</h6>'+
          '<a href="'+lastpart+'" target="blank">'+'<img style="padding-top: 25px;" src="/congress_pics/'+x+'.jpg" class="img-fluid rounded mx-auto d-block">'+'</a>'+
        '</div>'+
          '<div class="col-md-7">'+
            '<h1>'+name+'</h1>'+
              '<p></p>'+
        '<h6>'+name+'</h6>'+
          '</div>'+
      '</div>'+
      '<div class="row">'+
        '<div class="col-md-11">'+
      '<h6>MORE INFO</h6>'+
          
        '</div>'+
      '</div>'
  
  )
 $('#load').css('display', 'none')
  $('#getlobbydata').fadeIn(20000).css('display', 'block')
          });
        });
    });
    // divmagic("#getlobbydata");
  });
  

  
  
  
  //Post method for influence interaction
  //parsed out bootstrap html with for loops for ajax data
  //picture with x url
  
  }
  
  //CARD SWITCH STATEMENT
    var q = '';
    var zz = function (q) {
      switch(q) {
              
              case 'Campaign Finance Reform':
              console.log("CF started"); 
              $("#lobby").css('display', 'none');
              $('#cards').css('display', 'none')
              $("#carousel").css('display', 'block');
              $("#caroucontent").css('display', 'block');
              window.localStorage.setItem('Path', 'D');
              var x2 = function () {
                //function for recalling issue specific API data and, after cross-referencing, developing location specific strategy of influence
              };
              x2;
              headerphoto1='Sanders Announces Second Presidential Bid';
              captionphoto1='Vermont Senator Bernie Sanders has announced his intention to seek 2020 Democratic Presidential Nomination.';
              linkphoto1='https://www.gannett-cdn.com/presto/2019/01/28/PDTN/0b1cf0e3-3286-40df-903f-353b35ae9240-AP_19010087848671.jpg?crop=2864,1604,x0,y37&width=3200&height=1680&fit=bounds';
              headerphoto2='Democratic Legislators seek reforms';
              captionphoto2='Speaker Pelosi has expressed interest in reforming what has become a very difficult system to navigate.';
              linkphoto2='';
              headerphoto3='Democratic Sun Launches New Platform';
              captionphoto3='Supreme court rulings on campaign finance regulations have caused rampant corruption on the Hill. The Democratic Sun has created a new tool to combat that corruption and give a voice back to the people.';
              linkphoto3='';
              var cstrap = {
               code: '<div class="column">' +
               '<div class="jumbotron text-center">'+
              '<h4 class="display">'+q+'</h4>'+
            '</div>'+
            '<div class="col-lg-12 text-center">' +
              '<div id="carouselExampleIndicators" class="carousel slide" data-ride="carousel" data-interval="10000">' +
                      '<ol class="carousel-indicators">'+
                        '<li data-target="#carouselExampleIndicators" data-slide-to="0" class="active"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="1"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="2"></li>' +
                      '</ol>' +
                      '<div class="carousel-inner">' +
                        '<div class="carousel-item active">' +
                              '<div class="carousel-caption d-none d-md-block">' +
                                      '<h5>'+headerphoto1+'</h5>' +
                                      '<p1>'+captionphoto1+'</p1>' +
                              '</div>' +
                          '<img class="d-block w-100" src="'+linkphoto1+'" alt="First slide">' +
                        '</div>' +
                        '<div class="carousel-item">' +
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto2+'</h5>'+
                                      '<p1>'+captionphoto2+'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto2+'" alt="Second slide">'+
                        '</div>'+
                        '<div class="carousel-item">'+
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto3+'</h5>'+
                                      '<p1>'+captionphoto3 +'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto3+'" alt="Third slide">' +
                        '</div>'+
                      '</div>'+
                      '<a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">'+
                        '<span class="carousel-control-prev-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Previous</span>'+
                      '</a>'+
                      '<a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">'+
                        '<span class="carousel-control-next-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Next</span>'+
                      '</a>'+
              '</div>'
              };
              var x1 = cstrap.code;
              document.getElementById("carousel").innerHTML = x1;
              var x = {
                rhetoric: "The influence of the People should not have to compete with the" + 
              " influence of corporate america. The weight of a checkbook shouldn't" + 
              " outweigh the common will. Donate now to start evening the playing field."
             };
              document.getElementById("caroucontent").innerHTML=x.rhetoric;
              $('#getlobbydata').delay(1000).fadeIn(3200);
              console.log("CF finished");
              break;
              
              case 'Build the Wall':
              $("#lobby").css('display', 'none');
              $("#carousel").css('display', 'block');
              $("#caroucontent").css('display', 'block');
              $('#cards').css('display', 'none')
              window.localStorage.setItem('Path', 'R');
              var y2 = function ()  {
                //function for recalling issue specific API data and, after cross-referencing, developing location specific strategy of influence
              };
              y2;
              headerphoto1='Local Man Decides to Proclaim Ignorance';
              captionphoto1='Town Dumbass hopeful decides to retire dick from further contact by "Flexing for Trump"';
              linkphoto1='http://www.trbimg.com/img-586f2ae4/turbine/ct-trump-congress-mexico-border-wall-20170105-001/750/750x422';
              headerphoto2='Rich Old White Men Rejoice';
              captionphoto2='Rich old dudes across the country are rejoicing at the chaotic and irrational nature a few racists rants can cause';
              linkphoto2='https://i3.wp.com/www.globalvillagespace.com/my_uploads/2017/02/trump-wall-min-640x358.png';
              headerphoto3='Local Developer Finally Says Fuck It';
              captionphoto3='Look I\'m not saying you guys are stupid I\'m just saying your all intellectually incapable of juxtaposing two contradictory ideas because you decided reading was "for the libs"';
              linkphoto3='https://ndpnews.org/wp-content/uploads/2016/11/mexico-fence-1024x455.jpg ';
              var cstrap = {
                code: '<div class="column">' +
                '<div class="jumbotron text-center">'+
               '<h4 class="display">'+q+'</h4>'+
             '</div>'+
             '<div class="col-lg-12 text-center">' +
               '<div id="carouselExampleIndicators" class="carousel slide" data-ride="carousel" data-interval="10000">' +
                       '<ol class="carousel-indicators">'+
                         '<li data-target="#carouselExampleIndicators" data-slide-to="0" class="active"></li>' +
                         '<li data-target="#carouselExampleIndicators" data-slide-to="1"></li>' +
                         '<li data-target="#carouselExampleIndicators" data-slide-to="2"></li>' +
                       '</ol>' +
                       '<div class="carousel-inner">' +
                         '<div class="carousel-item active">' +
                               '<div class="carousel-caption d-none d-md-block">' +
                                       '<h5>'+headerphoto1+'</h5>' +
                                       '<p1>'+captionphoto1+'</p1>' +
                               '</div>' +
                           '<img class="d-block w-100" src="'+linkphoto1+'" alt="First slide">' +
                         '</div>' +
                         '<div class="carousel-item">' +
                               '<div class="carousel-caption d-none d-md-block">'+
                                       '<h5>'+headerphoto2+'</h5>'+
                                       '<p1>'+captionphoto2+'</p1>'+
                               '</div>'+
                           '<img class="d-block w-100" src="'+linkphoto2+'" alt="Second slide">'+
                         '</div>'+
                         '<div class="carousel-item">'+
                               '<div class="carousel-caption d-none d-md-block">'+
                                       '<h5>'+headerphoto3+'</h5>'+
                                       '<p1>'+captionphoto3 +'</p1>'+
                               '</div>'+
                           '<img class="d-block w-100" src="'+linkphoto3+'" alt="Third slide">' +
                         '</div>'+
                       '</div>'+
                       '<a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">'+
                         '<span class="carousel-control-prev-icon" aria-hidden="true"></span>'+
                         '<span class="sr-only">Previous</span>'+
                       '</a>'+
                       '<a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">'+
                         '<span class="carousel-control-next-icon" aria-hidden="true"></span>'+
                         '<span class="sr-only">Next</span>'+
                       '</a>'+
               '</div>'
               };
               var y1 = cstrap.code;
              document.getElementById("carousel").innerHTML=y1;
              var y ={
              rhetoric: "BUT THE CARAVANS"
              };
              document.getElementById("caroucontent").innerHTML=y.rhetoric;
              $('#getlobbydata').delay(2000).fadeIn(3200);
              break;
              
              case 'Medicare for All':
              $("#lobby").css('display', 'none');
              $("#carousel").css('display', 'block');
              $("#caroucontent").css('display', 'block');
              $('#cards').css('display', 'none');
              window.localStorage.setItem('Path', 'D');
              var z2 = function () {
                //function for recalling issue specific API data and, after cross-referencing, developing location specific strategy of influence
              };
              z2;
              headerphoto1='New Study Finds Startling Data';
              captionphoto1='A new study published by the Duke Medical Center has found that not killing people is a total bro move.';
              linkphoto1='https://www.jpl.nasa.gov/edu/images/activities/temp_graphing_g5_step11.jpg';
              headerphoto2='Aids Related Deaths Curbed Dramatically';
              captionphoto2='It turns out by providing universal healthcare most diseases can be eradicated due to herd immunity..That sounds smart';
              linkphoto2='https://i.ytimg.com/vi/dscxMj7DtGc/maxresdefault.jpg';
              headerphoto3='Skiing Related Injuries Unaffected';
              captionphoto3='Unfortunately for the nations elite, traversing the nations slopes plastered on whatever wine pairs well with their $900 entree has been unaffected by nationwide coverage.';
              linkphoto3='https://usatftw.files.wordpress.com/2017/05/spongebob.jpg?w=1000&h=600&crop=1';
              var cstrap = {
                code: '<div class="column">' +
                '<div class="jumbotron text-center">'+
               '<h4 class="display">'+q+'</h4>'+
             '</div>'+
             '<div class="col-lg-12 text-center">' +
               '<div id="carouselExampleIndicators" class="carousel slide" data-ride="carousel" data-interval="10000">' +
                       '<ol class="carousel-indicators">'+
                         '<li data-target="#carouselExampleIndicators" data-slide-to="0" class="active"></li>' +
                         '<li data-target="#carouselExampleIndicators" data-slide-to="1"></li>' +
                         '<li data-target="#carouselExampleIndicators" data-slide-to="2"></li>' +
                       '</ol>' +
                       '<div class="carousel-inner">' +
                         '<div class="carousel-item active">' +
                               '<div class="carousel-caption d-none d-md-block">' +
                                       '<h5>'+headerphoto1+'</h5>' +
                                       '<p1>'+captionphoto1+'</p1>' +
                               '</div>' +
                           '<img class="d-block w-100" src="'+linkphoto1+'" alt="First slide">' +
                         '</div>' +
                         '<div class="carousel-item">' +
                               '<div class="carousel-caption d-none d-md-block">'+
                                       '<h5>'+headerphoto2+'</h5>'+
                                       '<p1>'+captionphoto2+'</p1>'+
                               '</div>'+
                           '<img class="d-block w-100" src="'+linkphoto2+'" alt="Second slide">'+
                         '</div>'+
                         '<div class="carousel-item">'+
                               '<div class="carousel-caption d-none d-md-block">'+
                                       '<h5>'+headerphoto3+'</h5>'+
                                       '<p1>'+captionphoto3 +'</p1>'+
                               '</div>'+
                           '<img class="d-block w-100" src="'+linkphoto3+'" alt="Third slide">' +
                         '</div>'+
                       '</div>'+
                       '<a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">'+
                         '<span class="carousel-control-prev-icon" aria-hidden="true"></span>'+
                         '<span class="sr-only">Previous</span>'+
                       '</a>'+
                       '<a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">'+
                         '<span class="carousel-control-next-icon" aria-hidden="true"></span>'+
                         '<span class="sr-only">Next</span>'+
                       '</a>'+
               '</div>'
               };
               var z1= cstrap.code;
              document.getElementById("carousel").innerHTML=z1;
              var z ={ 
              rhetoric: "Too bad you can't immunize yourself against these bomb studies loser" 
            }
              document.getElementById("caroucontent").innerHTML=z.rhetoric;
              $('#getlobbydata').delay(2000).fadeIn(3200);
              break;
              
              case 'Gun Rights':
              $("#lobby").css('display', 'none');
              $("#carousel").css('display', 'block');
              $("#caroucontent").css('display', 'block');
              $('#cards').css('display', 'none');
              window.localStorage.setItem('Path', 'R');
              var a2 = function () {
                //function for recalling issue specific API data and, after cross-referencing, developing location specific strategy of influence
              };
              a2;
              headerphoto1='';
              captionphoto1='';
              linkphoto1='';
              headerphoto2='';
              captionphoto2='';
              linkphoto2='';
              headerphoto3='';
              captionphoto3='';
              linkphoto3='';
              var cstrap = {
                code: '<div class="column">' +
                '<div class="jumbotron text-center">'+
               '<h4 class="display">'+q+'</h4>'+
             '</div>'+
             '<div class="col-lg-12 text-center">' +
               '<div id="carouselExampleIndicators" class="carousel slide" data-ride="carousel" data-interval="10000">' +
                       '<ol class="carousel-indicators">'+
                         '<li data-target="#carouselExampleIndicators" data-slide-to="0" class="active"></li>' +
                         '<li data-target="#carouselExampleIndicators" data-slide-to="1"></li>' +
                         '<li data-target="#carouselExampleIndicators" data-slide-to="2"></li>' +
                       '</ol>' +
                       '<div class="carousel-inner">' +
                         '<div class="carousel-item active">' +
                               '<div class="carousel-caption d-none d-md-block">' +
                                       '<h5>'+headerphoto1+'</h5>' +
                                       '<p1>'+captionphoto1+'</p1>' +
                               '</div>' +
                           '<img class="d-block w-100" src="'+linkphoto1+'" alt="First slide">' +
                         '</div>' +
                         '<div class="carousel-item">' +
                               '<div class="carousel-caption d-none d-md-block">'+
                                       '<h5>'+headerphoto2+'</h5>'+
                                       '<p1>'+captionphoto2+'</p1>'+
                               '</div>'+
                           '<img class="d-block w-100" src="'+linkphoto2+'" alt="Second slide">'+
                         '</div>'+
                         '<div class="carousel-item">'+
                               '<div class="carousel-caption d-none d-md-block">'+
                                       '<h5>'+headerphoto3+'</h5>'+
                                       '<p1>'+captionphoto3 +'</p1>'+
                               '</div>'+
                           '<img class="d-block w-100" src="'+linkphoto3+'" alt="Third slide">' +
                         '</div>'+
                       '</div>'+
                       '<a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">'+
                         '<span class="carousel-control-prev-icon" aria-hidden="true"></span>'+
                         '<span class="sr-only">Previous</span>'+
                       '</a>'+
                       '<a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">'+
                         '<span class="carousel-control-next-icon" aria-hidden="true"></span>'+
                         '<span class="sr-only">Next</span>'+
                       '</a>'+
               '</div>'
               };
               var a1 = cstrap.code;
              document.getElementById("carousel").innerHTML=a1;
              var a = {rhetoric: "Rhetoric"};
              document.getElementById("caroucontent").innerHTML=a.rhetoric;
              $('#getlobbydata').delay(2000).fadeIn(3200);
              break;
              
              case 'Free Collegiate Education':
              $("#lobby").css('display', 'none');
              $("#carousel").css('display', 'block');
              $("#caroucontent").css('display', 'block');
              $('#cards').css('display', 'none');
              window.localStorage.setItem('Path', 'D');
              var b2 = function () {
                //function for recalling issue specific API data and, after cross-referencing, developing location specific strategy of influence
              };
              b2;
              headerphoto1='Free Collegiate Education';
              captionphoto1='';
              linkphoto1='';
              headerphoto2='';
              captionphoto2='';
              linkphoto2='';
              headerphoto3='';
              captionphoto3='';
              linkphoto3='';
              var cstrap = {
               code: '<div class="column">' +
               '<div class="jumbotron text-center">'+
              '<h4 class="display">'+q+'</h4>'+
            '</div>'+
            '<div class="col-lg-12 text-center">' +
              '<div id="carouselExampleIndicators" class="carousel slide" data-ride="carousel" data-interval="10000">' +
                      '<ol class="carousel-indicators">'+
                        '<li data-target="#carouselExampleIndicators" data-slide-to="0" class="active"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="1"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="2"></li>' +
                      '</ol>' +
                      '<div class="carousel-inner">' +
                        '<div class="carousel-item active">' +
                              '<div class="carousel-caption d-none d-md-block">' +
                                      '<h5>'+headerphoto1+'</h5>' +
                                      '<p1>'+captionphoto1+'</p1>' +
                              '</div>' +
                          '<img class="d-block w-100" src="'+linkphoto1+'" alt="First slide">' +
                        '</div>' +
                        '<div class="carousel-item">' +
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto2+'</h5>'+
                                      '<p1>'+captionphoto2+'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto2+'" alt="Second slide">'+
                        '</div>'+
                        '<div class="carousel-item">'+
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto3+'</h5>'+
                                      '<p1>'+captionphoto3 +'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto3+'" alt="Third slide">' +
                        '</div>'+
                      '</div>'+
                      '<a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">'+
                        '<span class="carousel-control-prev-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Previous</span>'+
                      '</a>'+
                      '<a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">'+
                        '<span class="carousel-control-next-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Next</span>'+
                      '</a>'+
              '</div>'
              };
              var b1 = cstrap.code;
              document.getElementById("carousel").innerHTML=b1;
              var b = {rhetoric: "Rhetoric"};
              document.getElementById("caroucontent").innerHTML=b.rhetoric;
              $('#getlobbydata').delay(2000).fadeIn(3200);
              break;
              
              case 'Small Business Tax Reduction':
              $("#lobby").css('display', 'none');
              $("#carousel").css('display', 'block');
              $("#caroucontent").css('display', 'block');
              $('#cards').css('display', 'none');
              window.localStorage.setItem('Path','R');
              var c2 = function () {
                //function for recalling issue specific API data and, after cross-referencing, developing location specific strategy of influence
              };
              c2;
              headerphoto1='Small Business Tax Reduction';
              captionphoto1='';
              linkphoto1='';
              headerphoto2='';
              captionphoto2='';
              linkphoto2='';
              headerphoto3='';
              captionphoto3='';
              linkphoto3='';
              var cstrap = {
               code: '<div class="column">' +
               '<div class="jumbotron text-center">'+
              '<h4 class="display">'+q+'</h4>'+
            '</div>'+
            '<div class="col-lg-12 text-center">' +
              '<div id="carouselExampleIndicators" class="carousel slide" data-ride="carousel" data-interval="10000">' +
                      '<ol class="carousel-indicators">'+
                        '<li data-target="#carouselExampleIndicators" data-slide-to="0" class="active"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="1"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="2"></li>' +
                      '</ol>' +
                      '<div class="carousel-inner">' +
                        '<div class="carousel-item active">' +
                              '<div class="carousel-caption d-none d-md-block">' +
                                      '<h5>'+headerphoto1+'</h5>' +
                                      '<p1>'+captionphoto1+'</p1>' +
                              '</div>' +
                          '<img class="d-block w-100" src="'+linkphoto1+'" alt="First slide">' +
                        '</div>' +
                        '<div class="carousel-item">' +
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto2+'</h5>'+
                                      '<p1>'+captionphoto2+'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto2+'" alt="Second slide">'+
                        '</div>'+
                        '<div class="carousel-item">'+
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto3+'</h5>'+
                                      '<p1>'+captionphoto3 +'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto3+'" alt="Third slide">' +
                        '</div>'+
                      '</div>'+
                      '<a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">'+
                        '<span class="carousel-control-prev-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Previous</span>'+
                      '</a>'+
                      '<a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">'+
                        '<span class="carousel-control-next-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Next</span>'+
                      '</a>'+
              '</div>'
              };
              var c1 = cstrap.code;
              document.getElementById("carousel").innerHTML=c1;
              var c = {rhetoric: "Rhetoric"};
              document.getElementById("caroucontent").innerHTML=c.rhetoric;
              $('#getlobbydata').delay(2000).fadeIn(3200);
              break;
              
              case 'The Green New Deal':
              $("#lobby").css('display', 'none');
              $("#carousel").css('display', 'block');
              $("#caroucontent").css('display', 'block');
              $('#cards').css('display', 'none');
              window.localStorage.setItem('Path', 'D');
              var d2 = function () {
                //function for recalling issue specific API data and, after cross-referencing, developing location specific strategy of influence
              };
              d2;
              headerphoto1='The Green New Deal';
              captionphoto1='';
              linkphoto1='';
              headerphoto2='';
              captionphoto2='';
              linkphoto2='';
              headerphoto3='';
              captionphoto3='';
              linkphoto3='';
              var cstrap = {
               code: '<div class="column">' +
               '<div class="jumbotron text-center">'+
              '<h4 class="display">'+q+'</h4>'+
            '</div>'+
            '<div class="col-lg-12 text-center">' +
              '<div id="carouselExampleIndicators" class="carousel slide" data-ride="carousel" data-interval="10000">' +
                      '<ol class="carousel-indicators">'+
                        '<li data-target="#carouselExampleIndicators" data-slide-to="0" class="active"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="1"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="2"></li>' +
                      '</ol>' +
                      '<div class="carousel-inner">' +
                        '<div class="carousel-item active">' +
                              '<div class="carousel-caption d-none d-md-block">' +
                                      '<h5>'+headerphoto1+'</h5>' +
                                      '<p1>'+captionphoto1+'</p1>' +
                              '</div>' +
                          '<img class="d-block w-100" src="'+linkphoto1+'" alt="First slide">' +
                        '</div>' +
                        '<div class="carousel-item">' +
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto2+'</h5>'+
                                      '<p1>'+captionphoto2+'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto2+'" alt="Second slide">'+
                        '</div>'+
                        '<div class="carousel-item">'+
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto3+'</h5>'+
                                      '<p1>'+captionphoto3 +'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto3+'" alt="Third slide">' +
                        '</div>'+
                      '</div>'+
                      '<a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">'+
                        '<span class="carousel-control-prev-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Previous</span>'+
                      '</a>'+
                      '<a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">'+
                        '<span class="carousel-control-next-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Next</span>'+
                      '</a>'+
              '</div>'
              };
              var d1 = cstrap.code;
              document.getElementById("carousel").innerHTML=d1;
              var d = {rhetoric: "Rhetoric"};
              document.getElementById("caroucontent").innerHTML=d.rhetoric;
              $('#getlobbydata').delay(2000).fadeIn(3200);
              break;
              
              case 'Clean Coal Initiatives':
              $("#lobby").css('display', 'none');
              $("#carousel").css('display', 'block');
              $("#caroucontent").css('display', 'block');
              $('#cards').css('display', 'none');
              window.localStorage.setItem('Path', 'R');
              var e2 = function () {
                //function for recalling issue specific API data and, after cross-referencing, developing location specific strategy of influence
              };
              e2;
              headerphoto1='Clean Coal Initiatives';
              captionphoto1='';
              linkphoto1='';
              headerphoto2='';
              captionphoto2='';
              linkphoto2='';
              headerphoto3='';
              captionphoto3='';
              linkphoto3='';
              var cstrap = {
                code: '<div class="column">' +
                '<div class="jumbotron text-center">'+
               '<h4 class="display">'+q+'</h4>'+
             '</div>'+
             '<div class="col-lg-12 text-center">' +
               '<div id="carouselExampleIndicators" class="carousel slide" data-ride="carousel" data-interval="10000">' +
                       '<ol class="carousel-indicators">'+
                         '<li data-target="#carouselExampleIndicators" data-slide-to="0" class="active"></li>' +
                         '<li data-target="#carouselExampleIndicators" data-slide-to="1"></li>' +
                         '<li data-target="#carouselExampleIndicators" data-slide-to="2"></li>' +
                       '</ol>' +
                       '<div class="carousel-inner">' +
                         '<div class="carousel-item active">' +
                               '<div class="carousel-caption d-none d-md-block">' +
                                       '<h5>'+headerphoto1+'</h5>' +
                                       '<p1>'+captionphoto1+'</p1>' +
                               '</div>' +
                           '<img class="d-block w-100" src="'+linkphoto1+'" alt="First slide">' +
                         '</div>' +
                         '<div class="carousel-item">' +
                               '<div class="carousel-caption d-none d-md-block">'+
                                       '<h5>'+headerphoto2+'</h5>'+
                                       '<p1>'+captionphoto2+'</p1>'+
                               '</div>'+
                           '<img class="d-block w-100" src="'+linkphoto2+'" alt="Second slide">'+
                         '</div>'+
                         '<div class="carousel-item">'+
                               '<div class="carousel-caption d-none d-md-block">'+
                                       '<h5>'+headerphoto3+'</h5>'+
                                       '<p1>'+captionphoto3 +'</p1>'+
                               '</div>'+
                           '<img class="d-block w-100" src="'+linkphoto3+'" alt="Third slide">' +
                         '</div>'+
                       '</div>'+
                       '<a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">'+
                         '<span class="carousel-control-prev-icon" aria-hidden="true"></span>'+
                         '<span class="sr-only">Previous</span>'+
                       '</a>'+
                       '<a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">'+
                         '<span class="carousel-control-next-icon" aria-hidden="true"></span>'+
                         '<span class="sr-only">Next</span>'+
                       '</a>'+
               '</div>'
               };
               var e1 = cstrap.code;
              document.getElementById("carousel").innerHTML=e1;
              var e = {rhetoric: "Rhetoric"};
              document.getElementById("caroucontent").innerHTML=e.rhetoric;
              $('#getlobbydata').delay(2000).fadeIn(3200);
              break;
              
              case 'Corporate Inversion Penalties':
              $("#lobby").css('display', 'none');
              $("#carousel").css('display', 'block');
              $("#caroucontent").css('display', 'block');
              $('#cards').css('display', 'none');
              window.localStorage.setItem('Path', 'D');
              var f2 = function () {
                //function for recalling issue specific API data and, after cross-referencing, developing location specific strategy of influence
              };
              f2;
              headerphoto1='Corporate Inversion Penalties';
              captionphoto1='';
              linkphoto1='';
              headerphoto2='';
              captionphoto2='';
              linkphoto2='';
              headerphoto3='';
              captionphoto3='';
              linkphoto3='';
              var cstrap = {
               code: '<div class="column">' +
               '<div class="jumbotron text-center">'+
              '<h4 class="display">'+q+'</h4>'+
            '</div>'+
            '<div class="col-lg-12 text-center">' +
              '<div id="carouselExampleIndicators" class="carousel slide" data-ride="carousel" data-interval="10000">' +
                      '<ol class="carousel-indicators">'+
                        '<li data-target="#carouselExampleIndicators" data-slide-to="0" class="active"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="1"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="2"></li>' +
                      '</ol>' +
                      '<div class="carousel-inner">' +
                        '<div class="carousel-item active">' +
                              '<div class="carousel-caption d-none d-md-block">' +
                                      '<h5>'+headerphoto1+'</h5>' +
                                      '<p1>'+captionphoto1+'</p1>' +
                              '</div>' +
                          '<img class="d-block w-100" src="'+linkphoto1+'" alt="First slide">' +
                        '</div>' +
                        '<div class="carousel-item">' +
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto2+'</h5>'+
                                      '<p1>'+captionphoto2+'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto2+'" alt="Second slide">'+
                        '</div>'+
                        '<div class="carousel-item">'+
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto3+'</h5>'+
                                      '<p1>'+captionphoto3 +'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto3+'" alt="Third slide">' +
                        '</div>'+
                      '</div>'+
                      '<a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">'+
                        '<span class="carousel-control-prev-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Previous</span>'+
                      '</a>'+
                      '<a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">'+
                        '<span class="carousel-control-next-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Next</span>'+
                      '</a>'+
              '</div>'
              };
              var f1 = cstrap.code;
              document.getElementById("carousel").innerHTML=f1;
              var f = {rhetoric: "Rhetoric"};
              document.getElementById("caroucontent").innerHTML=f.rhetoric;
              $('#getlobbydata').delay(2000).fadeIn(3200);
              break;
              
              case 'Corporate Tax Reductions':
              $("#lobby").css('display', 'none');
              $("#carousel").css('display', 'block');
              $("#caroucontent").css('display', 'block');
              $('#cards').css('display', 'none');
              window.localStorage.setItem('Path', 'R');
              var g2 = function () {
                //function for recalling issue specific API data and, after cross-referencing, developing location specific strategy of influence
              };
              g2;
              headerphoto1='Corporate Tax Reductions';
              captionphoto1='';
              linkphoto1='';
              headerphoto2='';
              captionphoto2='';
              linkphoto2='';
              headerphoto3='';
              captionphoto3='';
              linkphoto3='';
              var cstrap = {
               code: '<div class="column">' +
               '<div class="jumbotron text-center">'+
              '<h4 class="display">'+q+'</h4>'+
            '</div>'+
            '<div class="col-lg-12 text-center">' +
              '<div id="carouselExampleIndicators" class="carousel slide" data-ride="carousel" data-interval="10000">' +
                      '<ol class="carousel-indicators">'+
                        '<li data-target="#carouselExampleIndicators" data-slide-to="0" class="active"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="1"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="2"></li>' +
                      '</ol>' +
                      '<div class="carousel-inner">' +
                        '<div class="carousel-item active">' +
                              '<div class="carousel-caption d-none d-md-block">' +
                                      '<h5>'+headerphoto1+'</h5>' +
                                      '<p1>'+captionphoto1+'</p1>' +
                              '</div>' +
                          '<img class="d-block w-100" src="'+linkphoto1+'" alt="First slide">' +
                        '</div>' +
                        '<div class="carousel-item">' +
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto2+'</h5>'+
                                      '<p1>'+captionphoto2+'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto2+'" alt="Second slide">'+
                        '</div>'+
                        '<div class="carousel-item">'+
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto3+'</h5>'+
                                      '<p1>'+captionphoto3 +'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto3+'" alt="Third slide">' +
                        '</div>'+
                      '</div>'+
                      '<a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">'+
                        '<span class="carousel-control-prev-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Previous</span>'+
                      '</a>'+
                      '<a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">'+
                        '<span class="carousel-control-next-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Next</span>'+
                      '</a>'+
              '</div>'
              };
              var g1 = cstrap.code;
              document.getElementById("carousel").innerHTML=g1;
              var g = {rhetoric: "Rhetoric"};
              document.getElementById("caroucontent").innerHTML=g.rhetoric;
              $('#getlobbydata').delay(2000).fadeIn(3200);
              break;
              
              case 'Abolish the Death Penalty':
              $("#lobby").css('display', 'none');
              $("#carousel").css('display', 'block');
              $("#caroucontent").css('display', 'block');
              $('#cards').css('display', 'none');
              window.localStorage.setItem('Path', 'D');
              var h2 = function () {
                //function for recalling issue specific API data and, after cross-referencing, developing location specific strategy of influence.
              };
              h2;
              headerphoto1='Abolish the Death Penalty';
              captionphoto1='';
              linkphoto1='';
              headerphoto2='';
              captionphoto2='';
              linkphoto2='';
              headerphoto3='';
              captionphoto3='';
              linkphoto3='';
              var cstrap = {
                code: '<div class="column">' +
                '<div class="jumbotron text-center">'+
               '<h4 class="display">'+q+'</h4>'+
             '</div>'+
             '<div class="col-lg-12 text-center">' +
               '<div id="carouselExampleIndicators" class="carousel slide" data-ride="carousel" data-interval="10000">' +
                       '<ol class="carousel-indicators">'+
                         '<li data-target="#carouselExampleIndicators" data-slide-to="0" class="active"></li>' +
                         '<li data-target="#carouselExampleIndicators" data-slide-to="1"></li>' +
                         '<li data-target="#carouselExampleIndicators" data-slide-to="2"></li>' +
                       '</ol>' +
                       '<div class="carousel-inner">' +
                         '<div class="carousel-item active">' +
                               '<div class="carousel-caption d-none d-md-block">' +
                                       '<h5>'+headerphoto1+'</h5>' +
                                       '<p1>'+captionphoto1+'</p1>' +
                               '</div>' +
                           '<img class="d-block w-100" src="'+linkphoto1+'" alt="First slide">' +
                         '</div>' +
                         '<div class="carousel-item">' +
                               '<div class="carousel-caption d-none d-md-block">'+
                                       '<h5>'+headerphoto2+'</h5>'+
                                       '<p1>'+captionphoto2+'</p1>'+
                               '</div>'+
                           '<img class="d-block w-100" src="'+linkphoto2+'" alt="Second slide">'+
                         '</div>'+
                         '<div class="carousel-item">'+
                               '<div class="carousel-caption d-none d-md-block">'+
                                       '<h5>'+headerphoto3+'</h5>'+
                                       '<p1>'+captionphoto3 +'</p1>'+
                               '</div>'+
                           '<img class="d-block w-100" src="'+linkphoto3+'" alt="Third slide">' +
                         '</div>'+
                       '</div>'+
                       '<a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">'+
                         '<span class="carousel-control-prev-icon" aria-hidden="true"></span>'+
                         '<span class="sr-only">Previous</span>'+
                       '</a>'+
                       '<a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">'+
                         '<span class="carousel-control-next-icon" aria-hidden="true"></span>'+
                         '<span class="sr-only">Next</span>'+
                       '</a>'+
               '</div>'
               };
               var h1 = cstrap.code;
              document.getElementById("carousel").innerHTML=h1;
              var h ={rhetoric: "rhetoric"};
              document.getElementById("caroucontent").innerHTML=h.rhetoric;
              $('#getlobbydata').delay(2000).fadeIn(3200);
              break;
  
              case 'Pro Death Penalty':
              $("#lobby").css('display', 'none');
              $("#carousel").css('display', 'block');
              $("#caroucontent").css('display', 'block');
              $('#cards').css('display', 'none');
              window.localStorage.setItem('Path', 'R');
              var i2 = function () {
                //function for recalling issue specific API data and, after cross-referencing, developing location specific strategy of influence
              };
              i2;
              headerphoto1='Pro Death Penalty';
              captionphoto1='';
              linkphoto1='';
              headerphoto2='';
              captionphoto2='';
              linkphoto2='';
              headerphoto3='';
              captionphoto3='';
              linkphoto3='';
              var cstrap = {
               code: '<div class="column">' +
               '<div class="jumbotron text-center">'+
              '<h4 class="display">'+q+'</h4>'+
            '</div>'+
            '<div class="col-lg-12 text-center">' +
              '<div id="carouselExampleIndicators" class="carousel slide" data-ride="carousel" data-interval="10000">' +
                      '<ol class="carousel-indicators">'+
                        '<li data-target="#carouselExampleIndicators" data-slide-to="0" class="active"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="1"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="2"></li>' +
                      '</ol>' +
                      '<div class="carousel-inner">' +
                        '<div class="carousel-item active">' +
                              '<div class="carousel-caption d-none d-md-block">' +
                                      '<h5>'+headerphoto1+'</h5>' +
                                      '<p1>'+captionphoto1+'</p1>' +
                              '</div>' +
                          '<img class="d-block w-100" src="'+linkphoto1+'" alt="First slide">' +
                        '</div>' +
                        '<div class="carousel-item">' +
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto2+'</h5>'+
                                      '<p1>'+captionphoto2+'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto2+'" alt="Second slide">'+
                        '</div>'+
                        '<div class="carousel-item">'+
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto3+'</h5>'+
                                      '<p1>'+captionphoto3 +'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto3+'" alt="Third slide">' +
                        '</div>'+
                      '</div>'+
                      '<a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">'+
                        '<span class="carousel-control-prev-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Previous</span>'+
                      '</a>'+
                      '<a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">'+
                        '<span class="carousel-control-next-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Next</span>'+
                      '</a>'+
              '</div>'
              };
              var i1 = cstrap.code;
              document.getElementById("carousel").innerHTML=i1;
              var i ={ rhetoric:"Rhetoric"};
              document.getElementById("caroucontent").innerHTML=i.rhetoric;
              $('#getlobbydata').delay(2000).fadeIn(3200);
              break;
              
              case 'Seperate Church and State':
              $("#lobby").css('display', 'none');
              $("#carousel").css('display', 'block');
              $("#caroucontent").css('display', 'block');
              $('#cards').css('display', 'none');
              window.localStorage.setItem('Path', 'D');
              var j2 = function () {
                //function for recalling issue specific API data and, after cross-referencing, developing location specific strategy of influence
              };
              j2;
              headerphoto1='Seperate Church and State';
              captionphoto1='';
              linkphoto1='';
              headerphoto2='';
              captionphoto2='';
              linkphoto2='';
              headerphoto3='';
              captionphoto3='';
              linkphoto3='';
              var cstrap = {
               code: '<div class="column">' +
               '<div class="jumbotron text-center">'+
              '<h4 class="display">'+q+'</h4>'+
            '</div>'+
            '<div class="col-lg-12 text-center">' +
              '<div id="carouselExampleIndicators" class="carousel slide" data-ride="carousel" data-interval="10000">' +
                      '<ol class="carousel-indicators">'+
                        '<li data-target="#carouselExampleIndicators" data-slide-to="0" class="active"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="1"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="2"></li>' +
                      '</ol>' +
                      '<div class="carousel-inner">' +
                        '<div class="carousel-item active">' +
                              '<div class="carousel-caption d-none d-md-block">' +
                                      '<h5>'+headerphoto1+'</h5>' +
                                      '<p1>'+captionphoto1+'</p1>' +
                              '</div>' +
                          '<img class="d-block w-100" src="'+linkphoto1+'" alt="First slide">' +
                        '</div>' +
                        '<div class="carousel-item">' +
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto2+'</h5>'+
                                      '<p1>'+captionphoto2+'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto2+'" alt="Second slide">'+
                        '</div>'+
                        '<div class="carousel-item">'+
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto3+'</h5>'+
                                      '<p1>'+captionphoto3 +'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto3+'" alt="Third slide">' +
                        '</div>'+
                      '</div>'+
                      '<a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">'+
                        '<span class="carousel-control-prev-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Previous</span>'+
                      '</a>'+
                      '<a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">'+
                        '<span class="carousel-control-next-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Next</span>'+
                      '</a>'+
              '</div>'
              };
              var j1 = cstrap.code;
              document.getElementById("carousel").innerHTML=j1;
              var j ={ rhetoric: "Rhetoric"};
              document.getElementById("caroucontent").innerHTML=j.rhetoric;
              $('#getlobbydata').delay(2000).fadeIn(3200);
              break;
              
              case 'Religious Expression':
              $("#lobby").css('display', 'none');
              $("#carousel").css('display', 'block');
              $("#caroucontent").css('display', 'block');
              $('#cards').css('display', 'none');
              window.localStorage.setItem('Path', 'R');
              var l2 = function () {
                //function for recalling issue specific API data and, after cross-referencing, developing location specific strategy of influence
              };
              l2;
              headerphoto1='Religious Expression';
              captionphoto1='';
              linkphoto1='';
              headerphoto2='';
              captionphoto2='';
              linkphoto2='';
              headerphoto3='';
              captionphoto3='';
              linkphoto3='';
              var cstrap = {
               code: '<div class="column">' +
               '<div class="jumbotron text-center">'+
              '<h4 class="display">'+q+'</h4>'+
            '</div>'+
            '<div class="col-lg-12 text-center">' +
              '<div id="carouselExampleIndicators" class="carousel slide" data-ride="carousel" data-interval="10000">' +
                      '<ol class="carousel-indicators">'+
                        '<li data-target="#carouselExampleIndicators" data-slide-to="0" class="active"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="1"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="2"></li>' +
                      '</ol>' +
                      '<div class="carousel-inner">' +
                        '<div class="carousel-item active">' +
                              '<div class="carousel-caption d-none d-md-block">' +
                                      '<h5>'+headerphoto1+'</h5>' +
                                      '<p1>'+captionphoto1+'</p1>' +
                              '</div>' +
                          '<img class="d-block w-100" src="'+linkphoto1+'" alt="First slide">' +
                        '</div>' +
                        '<div class="carousel-item">' +
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto2+'</h5>'+
                                      '<p1>'+captionphoto2+'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto2+'" alt="Second slide">'+
                        '</div>'+
                        '<div class="carousel-item">'+
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto3+'</h5>'+
                                      '<p1>'+captionphoto3 +'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto3+'" alt="Third slide">' +
                        '</div>'+
                      '</div>'+
                      '<a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">'+
                        '<span class="carousel-control-prev-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Previous</span>'+
                      '</a>'+
                      '<a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">'+
                        '<span class="carousel-control-next-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Next</span>'+
                      '</a>'+
              '</div>'
              };
              var l1 = cstrap.code;
              document.getElementById("carousel").innerHTML=l1;
              var l = {rhetoric:"Rhetoric"};
              document.getElementById("caroucontent").innerHTML=l.rhetoric;
              $('#getlobbydata').delay(2000).fadeIn(3200);
              break;
              
              case 'Same-Sex Marriage':
              $("#lobby").css('display', 'none');
              $("#carousel").css('display', 'block');
              $("#caroucontent").css('display', 'block');
              $('#cards').css('display', 'none');
              window.localStorage.setItem('Path', 'D');
              var m2 = function () {
                //function for recalling issue specific API data and, after cross-referencing, developing location specific strategy of influence
              };
              m2;
              headerphoto1='Same-Sex Marriage';
              captionphoto1='';
              linkphoto1='';
              headerphoto2='';
              captionphoto2='';
              linkphoto2='';
              headerphoto3='';
              captionphoto3='';
              linkphoto3='';
              var cstrap = {
               code: '<div class="column">' +
               '<div class="jumbotron text-center">'+
              '<h4 class="display">'+q+'</h4>'+
            '</div>'+
            '<div class="col-lg-12 text-center">' +
              '<div id="carouselExampleIndicators" class="carousel slide" data-ride="carousel" data-interval="10000">' +
                      '<ol class="carousel-indicators">'+
                        '<li data-target="#carouselExampleIndicators" data-slide-to="0" class="active"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="1"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="2"></li>' +
                      '</ol>' +
                      '<div class="carousel-inner">' +
                        '<div class="carousel-item active">' +
                              '<div class="carousel-caption d-none d-md-block">' +
                                      '<h5>'+headerphoto1+'</h5>' +
                                      '<p1>'+captionphoto1+'</p1>' +
                              '</div>' +
                          '<img class="d-block w-100" src="'+linkphoto1+'" alt="First slide">' +
                        '</div>' +
                        '<div class="carousel-item">' +
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto2+'</h5>'+
                                      '<p1>'+captionphoto2+'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto2+'" alt="Second slide">'+
                        '</div>'+
                        '<div class="carousel-item">'+
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto3+'</h5>'+
                                      '<p1>'+captionphoto3 +'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto3+'" alt="Third slide">' +
                        '</div>'+
                      '</div>'+
                      '<a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">'+
                        '<span class="carousel-control-prev-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Previous</span>'+
                      '</a>'+
                      '<a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">'+
                        '<span class="carousel-control-next-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Next</span>'+
                      '</a>'+
              '</div>'
              };
              var m1 = cstrap.code;
              document.getElementById("carousel").innerHTML=m1;
              var m = {rhetoric: "Rhetoric"};
              document.getElementById("caroucontent").innerHTML=m.rhetoric;
              $('#getlobbydata').delay(2000).fadeIn(3200);
              break;
              
              case 'The Sanctity of Marriage':
              $("#lobby").css('display', 'none');
              $("#carousel").css('display', 'block');
              $("#caroucontent").css('display', 'block');
              $('#cards').css('display', 'none');
              window.localStorage.setItem('Path', 'R');
              var n2 = function () {
                //function for recalling issue specific API data and, after cross-referencing, developing location specific strategy of influence
              };
              n2;
              headerphoto1='The Sanctity of Marriage';
              captionphoto1='';
              linkphoto1='';
              headerphoto2='';
              captionphoto2='';
              linkphoto2='';
              headerphoto3='';
              captionphoto3='';
              linkphoto3='';
              var cstrap = {
               code: '<div class="column">' +
               '<div class="jumbotron text-center">'+
              '<h4 class="display">'+q+'</h4>'+
            '</div>'+
            '<div class="col-lg-12 text-center">' +
              '<div id="carouselExampleIndicators" class="carousel slide" data-ride="carousel" data-interval="10000">' +
                      '<ol class="carousel-indicators">'+
                        '<li data-target="#carouselExampleIndicators" data-slide-to="0" class="active"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="1"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="2"></li>' +
                      '</ol>' +
                      '<div class="carousel-inner">' +
                        '<div class="carousel-item active">' +
                              '<div class="carousel-caption d-none d-md-block">' +
                                      '<h5>'+headerphoto1+'</h5>' +
                                      '<p1>'+captionphoto1+'</p1>' +
                              '</div>' +
                          '<img class="d-block w-100" src="'+linkphoto1+'" alt="First slide">' +
                        '</div>' +
                        '<div class="carousel-item">' +
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto2+'</h5>'+
                                      '<p1>'+captionphoto2+'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto2+'" alt="Second slide">'+
                        '</div>'+
                        '<div class="carousel-item">'+
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto3+'</h5>'+
                                      '<p1>'+captionphoto3 +'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto3+'" alt="Third slide">' +
                        '</div>'+
                      '</div>'+
                      '<a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">'+
                        '<span class="carousel-control-prev-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Previous</span>'+
                      '</a>'+
                      '<a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">'+
                        '<span class="carousel-control-next-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Next</span>'+
                      '</a>'+
              '</div>'
              };
              var n1 = cstrap.code;
              document.getElementById("carousel").innerHTML=n1;
              var n = {rhetoric: "Rhetoric"};
              document.getElementById("caroucontent").innerHTML=n.rhetoric;
              $('#getlobbydata').delay(2000).fadeIn(3200);
              break;
              
              case 'Universal Basic Income':
              $("#lobby").css('display', 'none');
              $("#carousel").css('display', 'block');
              $("#caroucontent").css('display', 'block');
              $('#cards').css('display', 'none');
              window.localStorage.setItem('Path', 'D');
              var o2 = function () {
                //function for recalling issue specific API data and, after cross-referencing, developing location specific strategy of influence
              };
              o2;
              headerphoto1='Universal Basic Income';
              captionphoto1='';
              linkphoto1='';
              headerphoto2='';
              captionphoto2='';
              linkphoto2='';
              headerphoto3='';
              captionphoto3='';
              linkphoto3='';
              var cstrap = {
               code: '<div class="column">' +
               '<div class="jumbotron text-center">'+
              '<h4 class="display">'+q+'</h4>'+
            '</div>'+
            '<div class="col-lg-12 text-center">' +
              '<div id="carouselExampleIndicators" class="carousel slide" data-ride="carousel" data-interval="10000">' +
                      '<ol class="carousel-indicators">'+
                        '<li data-target="#carouselExampleIndicators" data-slide-to="0" class="active"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="1"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="2"></li>' +
                      '</ol>' +
                      '<div class="carousel-inner">' +
                        '<div class="carousel-item active">' +
                              '<div class="carousel-caption d-none d-md-block">' +
                                      '<h5>'+headerphoto1+'</h5>' +
                                      '<p1>'+captionphoto1+'</p1>' +
                              '</div>' +
                          '<img class="d-block w-100" src="'+linkphoto1+'" alt="First slide">' +
                        '</div>' +
                        '<div class="carousel-item">' +
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto2+'</h5>'+
                                      '<p1>'+captionphoto2+'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto2+'" alt="Second slide">'+
                        '</div>'+
                        '<div class="carousel-item">'+
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto3+'</h5>'+
                                      '<p1>'+captionphoto3 +'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto3+'" alt="Third slide">' +
                        '</div>'+
                      '</div>'+
                      '<a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">'+
                        '<span class="carousel-control-prev-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Previous</span>'+
                      '</a>'+
                      '<a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">'+
                        '<span class="carousel-control-next-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Next</span>'+
                      '</a>'+
              '</div>'
              };
              var o1 = cstrap.code;
              document.getElementById("carousel").innerHTML=o1;
              var o = {rhetoric: "Rhetoric"};
              document.getElementById("caroucontent").innerHTML=o.rhetoric;
              $('#getlobbydata').delay(2000).fadeIn(3200);
              break;
              
              case 'The Welfare State':
              $("#lobby").css('display', 'none');
              $("#carousel").css('display', 'block');
              $("#caroucontent").css('display', 'block');
              $('#cards').css('display', 'none');
              window.localStorage.setItem('Path', 'R');
              var p2 = function () {
                //function for recalling issue specific API data and, after cross-referencing, developing location specific strategy of influence
              };
              p2;
              headerphoto1='The Welfare State';
              captionphoto1='';
              linkphoto1='';
              headerphoto2='';
              captionphoto2='';
              linkphoto2='';
              headerphoto3='';
              captionphoto3='';
              linkphoto3='';
              var cstrap = {
               code: '<div class="column">' +
               '<div class="jumbotron text-center">'+
              '<h4 class="display">'+q+'</h4>'+
            '</div>'+
            '<div class="col-lg-12 text-center">' +
              '<div id="carouselExampleIndicators" class="carousel slide" data-ride="carousel" data-interval="10000">' +
                      '<ol class="carousel-indicators">'+
                        '<li data-target="#carouselExampleIndicators" data-slide-to="0" class="active"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="1"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="2"></li>' +
                      '</ol>' +
                      '<div class="carousel-inner">' +
                        '<div class="carousel-item active">' +
                              '<div class="carousel-caption d-none d-md-block">' +
                                      '<h5>'+headerphoto1+'</h5>' +
                                      '<p1>'+captionphoto1+'</p1>' +
                              '</div>' +
                          '<img class="d-block w-100" src="'+linkphoto1+'" alt="First slide">' +
                        '</div>' +
                        '<div class="carousel-item">' +
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto2+'</h5>'+
                                      '<p1>'+captionphoto2+'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto2+'" alt="Second slide">'+
                        '</div>'+
                        '<div class="carousel-item">'+
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto3+'</h5>'+
                                      '<p1>'+captionphoto3 +'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto3+'" alt="Third slide">' +
                        '</div>'+
                      '</div>'+
                      '<a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">'+
                        '<span class="carousel-control-prev-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Previous</span>'+
                      '</a>'+
                      '<a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">'+
                        '<span class="carousel-control-next-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Next</span>'+
                      '</a>'+
              '</div>'
              };
              var p1 = cstrap.code;
              document.getElementById("carousel").innerHTML=p1;
              var p ={rhetoric: "Rhetoric"};
              document.getElementById("caroucontent").innerHTML=p.rhetoric;
              $('#getlobbydata').delay(2000).fadeIn(3200);
              break;
              
              case 'The War on Terror':
              $("#lobby").css('display', 'none');
              $("#carousel").css('display', 'block');
              $("#caroucontent").css('display', 'block');
              $('#cards').css('display', 'none');
              window.localStorage.setItem('Path', 'D');
              var r2 = function () {
                //function for recalling issue specific API data and, after cross-referencing, developing location specific strategy of influence
              };
              r2;
              headerphoto1='';
              captionphoto1='The War on Terror';
              linkphoto1='';
              headerphoto2='';
              captionphoto2='';
              linkphoto2='';
              headerphoto3='';
              captionphoto3='';
              linkphoto3='';
              var cstrap = {
               code: '<div class="column">' +
               '<div class="jumbotron text-center">'+
              '<h4 class="display">'+q+'</h4>'+
            '</div>'+
            '<div class="col-lg-12 text-center">' +
              '<div id="carouselExampleIndicators" class="carousel slide" data-ride="carousel" data-interval="10000">' +
                      '<ol class="carousel-indicators">'+
                        '<li data-target="#carouselExampleIndicators" data-slide-to="0" class="active"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="1"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="2"></li>' +
                      '</ol>' +
                      '<div class="carousel-inner">' +
                        '<div class="carousel-item active">' +
                              '<div class="carousel-caption d-none d-md-block">' +
                                      '<h5>'+headerphoto1+'</h5>' +
                                      '<p1>'+captionphoto1+'</p1>' +
                              '</div>' +
                          '<img class="d-block w-100" src="'+linkphoto1+'" alt="First slide">' +
                        '</div>' +
                        '<div class="carousel-item">' +
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto2+'</h5>'+
                                      '<p1>'+captionphoto2+'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto2+'" alt="Second slide">'+
                        '</div>'+
                        '<div class="carousel-item">'+
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto3+'</h5>'+
                                      '<p1>'+captionphoto3 +'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto3+'" alt="Third slide">' +
                        '</div>'+
                      '</div>'+
                      '<a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">'+
                        '<span class="carousel-control-prev-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Previous</span>'+
                      '</a>'+
                      '<a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">'+
                        '<span class="carousel-control-next-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Next</span>'+
                      '</a>'+
              '</div>'
              };
              var r1 = cstrap.code;
              document.getElementById("carousel").innerHTML=r1;
              var r = {rhetoric: "Rhetoric"};
              document.getElementById("caroucontent").innerHTML=r.rhetoric;
              $('#getlobbydata').delay(2000).fadeIn(3200);
              break;
              
              case 'America is Under-Attack':
              $("#lobby").css('display', 'none');
              $("#carousel").css('display', 'block');
              $("#caroucontent").css('display', 'block');
              $('#cards').css('display', 'none');
              window.localStorage.setItem('Path', 'R');
              var s2 = function () {
                //function for recalling issue specific API data and, after cross-referencing, developing location specific strategy of influence
              };
              s2;
              headerphoto1='';
              captionphoto1='America is Under-Attack';
              linkphoto1='';
              headerphoto2='';
              captionphoto2='';
              linkphoto2='';
              headerphoto3='';
              captionphoto3='';
              linkphoto3='';
              var cstrap = {
               code: '<div class="column">' +
               '<div class="jumbotron text-center">'+
              '<h4 class="display">'+q+'</h4>'+
            '</div>'+
            '<div class="col-lg-12 text-center">' +
              '<div id="carouselExampleIndicators" class="carousel slide" data-ride="carousel" data-interval="10000">' +
                      '<ol class="carousel-indicators">'+
                        '<li data-target="#carouselExampleIndicators" data-slide-to="0" class="active"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="1"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="2"></li>' +
                      '</ol>' +
                      '<div class="carousel-inner">' +
                        '<div class="carousel-item active">' +
                              '<div class="carousel-caption d-none d-md-block">' +
                                      '<h5>'+headerphoto1+'</h5>' +
                                      '<p1>'+captionphoto1+'</p1>' +
                              '</div>' +
                          '<img class="d-block w-100" src="'+linkphoto1+'" alt="First slide">' +
                        '</div>' +
                        '<div class="carousel-item">' +
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto2+'</h5>'+
                                      '<p1>'+captionphoto2+'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto2+'" alt="Second slide">'+
                        '</div>'+
                        '<div class="carousel-item">'+
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto3+'</h5>'+
                                      '<p1>'+captionphoto3 +'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto3+'" alt="Third slide">' +
                        '</div>'+
                      '</div>'+
                      '<a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">'+
                        '<span class="carousel-control-prev-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Previous</span>'+
                      '</a>'+
                      '<a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">'+
                        '<span class="carousel-control-next-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Next</span>'+
                      '</a>'+
              '</div>'
              };
              var s1 = cstrap.code;
              document.getElementById("carousel").innerHTML=s1;
              var s = {rhetoric: "Rhetoric"};
              document.getElementById("caroucontent").innerHTML=s.rhetoric;
              $('#getlobbydata').delay(2000).fadeIn(3200);
              break;
              
              case 'End Voter Suppression':
              $("#lobby").css('display', 'none');
              $("#carousel").css('display', 'block');
              $("#caroucontent").css('display', 'block');
              $('#cards').css('display', 'none');
              window.localStorage.setItem('Path', 'D');
              var t2 = function () {
                //function for recalling issue specific API data and, after cross-referencing, developing location specific strategy of influence
              };
              t2;
              headerphoto1='';
              captionphoto1='END VOTER SUPPRESSION';
              linkphoto1='';
              headerphoto2='';
              captionphoto2='';
              linkphoto2='';
              headerphoto3='';
              captionphoto3='';
              linkphoto3='';
              var cstrap = {
               code: '<div class="column">' +
               '<div class="jumbotron text-center">'+
              '<h4 class="display">'+q+'</h4>'+
            '</div>'+
            '<div class="col-lg-12 text-center">' +
              '<div id="carouselExampleIndicators" class="carousel slide" data-ride="carousel" data-interval="10000">' +
                      '<ol class="carousel-indicators">'+
                        '<li data-target="#carouselExampleIndicators" data-slide-to="0" class="active"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="1"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="2"></li>' +
                      '</ol>' +
                      '<div class="carousel-inner">' +
                        '<div class="carousel-item active">' +
                              '<div class="carousel-caption d-none d-md-block">' +
                                      '<h5>'+headerphoto1+'</h5>' +
                                      '<p1>'+captionphoto1+'</p1>' +
                              '</div>' +
                          '<img class="d-block w-100" src="'+linkphoto1+'" alt="First slide">' +
                        '</div>' +
                        '<div class="carousel-item">' +
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto2+'</h5>'+
                                      '<p1>'+captionphoto2+'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto2+'" alt="Second slide">'+
                        '</div>'+
                        '<div class="carousel-item">'+
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto3+'</h5>'+
                                      '<p1>'+captionphoto3 +'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto3+'" alt="Third slide">' +
                        '</div>'+
                      '</div>'+
                      '<a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">'+
                        '<span class="carousel-control-prev-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Previous</span>'+
                      '</a>'+
                      '<a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">'+
                        '<span class="carousel-control-next-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Next</span>'+
                      '</a>'+
              '</div>'
              };
              var t1 = cstrap.code;
              document.getElementById("carousel").innerHTML=t1;
              var t = {rhetoric: "Rhetoric"};
              document.getElementById("caroucontent").innerHTML=t.rhetoric;
              $('#getlobbydata').delay(2000).fadeIn(3200);
              break;
              
              case 'Stop Illegals from Voting':
              $("#lobby").css('display', 'none');
              $("#carousel").css('display', 'block');
              $("#caroucontent").css('display', 'block');
              $('#cards').css('display', 'none');
              window.localStorage.setItem('Path', 'R')
              var u2 = function () {
                //function for recalling issue specific API data and, after cross-referencing, developing location specific strategy of influence
              };
              u2;
              headerphoto1='';
              captionphoto1='';
              linkphoto1='';
              headerphoto2='';
              captionphoto2='';
              linkphoto2='';
              headerphoto3='';
              captionphoto3='';
              linkphoto3='';
              var cstrap = {
               code: '<div class="column">' +
               '<div class="jumbotron text-center">'+
              '<h4 class="display">'+q+'</h4>'+
            '</div>'+
            '<div class="col-lg-12 text-center">' +
              '<div id="carouselExampleIndicators" class="carousel slide" data-ride="carousel" data-interval="10000">' +
                      '<ol class="carousel-indicators">'+
                        '<li data-target="#carouselExampleIndicators" data-slide-to="0" class="active"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="1"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="2"></li>' +
                      '</ol>' +
                      '<div class="carousel-inner">' +
                        '<div class="carousel-item active">' +
                              '<div class="carousel-caption d-none d-md-block">' +
                                      '<h5>'+headerphoto1+'</h5>' +
                                      '<p1>'+captionphoto1+'</p1>' +
                              '</div>' +
                          '<img class="d-block w-100" src="'+linkphoto1+'" alt="First slide">' +
                        '</div>' +
                        '<div class="carousel-item">' +
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto2+'</h5>'+
                                      '<p1>'+captionphoto2+'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto2+'" alt="Second slide">'+
                        '</div>'+
                        '<div class="carousel-item">'+
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto3+'</h5>'+
                                      '<p1>'+captionphoto3 +'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto3+'" alt="Third slide">' +
                        '</div>'+
                      '</div>'+
                      '<a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">'+
                        '<span class="carousel-control-prev-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Previous</span>'+
                      '</a>'+
                      '<a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">'+
                        '<span class="carousel-control-next-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Next</span>'+
                      '</a>'+
              '</div>'
              };
              var u1 = cstrap.code;
              document.getElementById("carousel").innerHTML=u1;
              var u = {rhetoric: "Rhetoric"};
              document.getElementById("caroucontent").innerHTML=u.rhetoric;
              $('#getlobbydata').delay(2000).fadeIn(3200);
              break; 
              
              default:
              $("#lobby").css('display', 'none');
              $("#carousel").css('display', 'block');
              $("#caroucontent").css('display', 'block');
              $('#cards').css('display', 'none');
              window.localStorage.setItem('Path', 'Failure');
              q= "WHAT DID YOU DO";
              console.log("Card Pathway Failure");
              headerphoto1='Sweet Fancy Moses';
              captionphoto1='Local person breaks coolest thing to happen to the internet since people realized you could point cameras at cats. (Also yes good boy dog woof bark yes) ';
              linkphoto1='https://media.giphy.com/media/Pk20jMIe44bHa/giphy.gif';
              headerphoto2='Shame brought to local persons family';
              captionphoto2='Shame today ladies and gents. Pure shame. Bill Murray shame was brought down upon a local family after someone broke the DS all but ensuring the apocalypse';
              linkphoto2='https://media.giphy.com/media/jEkFtJMJuNTgc/giphy.gif';
              headerphoto3='Christmas cancelled after reckless actions of local internet consumer';
              captionphoto3='Santa just called us and said your actions are reprehensable. With Christmas out the window he will now consult with the Easter bunny about a possible withdrawal';
              linkphoto3='https://heavyeditorial.files.wordpress.com/2018/12/GettyImages-1074396864.jpg?quality=65&strip=all';
              var cstrap = {
               code: '<div class="column">' +
               '<div class="jumbotron text-center">'+
              '<h4 class="display">'+q+'</h4>'+
            '</div>'+
            '<div class="col-lg-12 text-center">' +
              '<div id="carouselExampleIndicators" class="carousel slide" data-ride="carousel" data-interval="10000">' +
                      '<ol class="carousel-indicators">'+
                        '<li data-target="#carouselExampleIndicators" data-slide-to="0" class="active"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="1"></li>' +
                        '<li data-target="#carouselExampleIndicators" data-slide-to="2"></li>' +
                      '</ol>' +
                      '<div class="carousel-inner">' +
                        '<div class="carousel-item active">' +
                              '<div class="carousel-caption d-none d-md-block">' +
                                      '<h5>'+headerphoto1+'</h5>' +
                                      '<p1>'+captionphoto1+'</p1>' +
                              '</div>' +
                          '<img class="d-block w-100" src="'+linkphoto1+'" alt="First slide">' +
                        '</div>' +
                        '<div class="carousel-item">' +
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto2+'</h5>'+
                                      '<p1>'+captionphoto2+'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto2+'" alt="Second slide">'+
                        '</div>'+
                        '<div class="carousel-item">'+
                              '<div class="carousel-caption d-none d-md-block">'+
                                      '<h5>'+headerphoto3+'</h5>'+
                                      '<p1>'+captionphoto3 +'</p1>'+
                              '</div>'+
                          '<img class="d-block w-100" src="'+linkphoto3+'" alt="Third slide">' +
                        '</div>'+
                      '</div>'+
                      '<a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-slide="prev">'+
                        '<span class="carousel-control-prev-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Previous</span>'+
                      '</a>'+
                      '<a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-slide="next">'+
                        '<span class="carousel-control-next-icon" aria-hidden="true"></span>'+
                        '<span class="sr-only">Next</span>'+
                      '</a>'+
              '</div>'
              };
              var u1 = cstrap.code;
              document.getElementById("carousel").innerHTML=u1;
              var u = {
                rhetoric: "Sorry folks. It looks like we've had some trouble accessing" +
                " your data. Try heading back to the previous page and selecting your issue again" +
                " or file a report and we'll look into it as soon as possible" +
                ". Thanks again for visiting and helping us in the fight against big money in politics. ",
                sarcasm: "Also, we know what the error was now. It was your fault. Nice job. Thanks for playing along."
  
            };
              document.getElementById("content").innerHTML=u.rhetoric;
          }
        }
  