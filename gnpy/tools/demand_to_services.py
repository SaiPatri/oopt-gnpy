"""
Convert high level demands into static requests for each type of mode and divide the demands equally amongst
the services
"""
import pandas as pd
import logging
import json
import numpy as np

_logger = logging.getLogger(__name__)


class DemandToServices:

    def __init__(self, demandFileName,service_filename, services_channel_spacing):
        """
        import the demand json, create the service json. Convert demand json to gnpy compatible service json.
        NEXT STEP: CONVERT ADVA network planner network directly to either gnpy json or HeCSON json.
        :param demandFileName: Exact location of demandFile
        """
        self.demandFileName = demandFileName
        self.service_filename = service_filename
        self.demands = pd.read_json(self.demandFileName)
        self.ch_spacing_list = np.array(services_channel_spacing)
        self.ch_spacing_list[::-1].sort()
        self.traffic_growth = np.array([0.15, 0.3, 0.54, 0.6, 0.3, 0.25, 0.22, 0.20, 0.18, 0.18])
        pass

    def convert_to_services(self, return_type, eol):
        """
        Converts higher level HeCSON demands into gnpy compatible services where each service has its own channel
        spacing. The total demand bitrate is for now equally divided amongst all "services". In HeCSON demands have only
         src, dest and bitrate attributes, but gnpy requires an additional "spacing" attribute and allocates trx only
        according to that spacing. Hence there is a need to convert high level demands into services with different
        spacing.
        HeCSON demands:
        a<->b 1000 Gbps
        gnpy services (for this demand):
        a<->b 37.5 GHz 250 Gbps
        a<->b 50 GHz 250 Gbps
        a<->b 75 GHz 250 Gbps
        a<->b 100 GHz 250 Gbps

        Note that higher bit rate is assigned to larger channel spacings.
        :param return_type :type string "dict" for pandas dataframe or "json" for location of json file
        :param eol :type boolean If EOL, considers EOL bitrate post traffic growth
        :return: services json file / dict
        """

        # TODO: Iterate over each demand
        services = {"path-request":[]}
        reqID = 0
        for key,demand in self.demands.iteritems():
            src = demand[0]
            dst = demand[1]
            yearly_bitrate = demand[2]
            if eol:
                final_bitrate = np.cumsum(yearly_bitrate*self.traffic_growth)[-1]
            else:
                final_bitrate = yearly_bitrate
            final_bitrate = final_bitrate * 1000000000.0

            for ch_spacing in self.ch_spacing_list:
                current_req = {"request-id": reqID, "source": "trx " + src, "destination": "trx " + dst}
                current_req["src-tp-id"] = current_req["source"]
                current_req["dst-tp-id"] = current_req["destination"]
                current_req["bidirectional"] = True
                current_req["path-constraints"] = {
                    "te-bandwidth":
                        {
                            "technology":"flexi-grid",
                            "trx_type": "adhoc",
                            "trx_mode": None,
                            "effective-freq-slot": [
                                {
                                    "N": "null",
                                    "M": "null"
                                }
                            ],
                            "spacing":0,
                            "max-nb-of-channel": None,
                            "output-power": 0.001958925417941673,
                            "path_bandwidth": 0
                        }
                }
                current_req["path-constraints"]["te-bandwidth"]["spacing"] = ch_spacing * 1000000000.0
                current_req["path-constraints"]["te-bandwidth"]["path_bandwidth"] = final_bitrate / len(self.ch_spacing_list)

                services["path-request"].append(current_req)
                reqID += 1

        if return_type == "dict":
            return services
        else:

            filename = "C:\\Users\\SaiP\\Documents\\gitProject\\oopt-gnpy\\gnpy\\example-data\\"+self.service_filename
            with open(filename,"w",encoding='utf-8') as outfile:
                json.dump(services,outfile,indent=4)
            return filename
