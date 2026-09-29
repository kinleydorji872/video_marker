class PresenterRubric:
    DIMENSIONS={"Message structure":"Clear opening, logical sequence, and conclusion.","Message clarity":"Ideas are expressed directly and understandably.","Evidence":"Claims are supported with examples, data, research, or experience.","Fluency":"Pace, pauses, filler words, and continuity are controlled.","Pronunciation":"Speech timing and clarity signals support understandable delivery.","Audience engagement":"The presentation connects ideas to the audience.","Confidence":"Delivery shows steady pace, vocal control, and preparation.","Visual quality":"Lighting, sharpness, framing, and camera stability support the presentation."}
    @classmethod
    def build(cls,breakdown):
        mapping={"Message structure":"message_structure","Message clarity":"message_clarity","Evidence":"evidence","Fluency":"fluency","Pronunciation":"pronunciation","Audience engagement":"audience_engagement","Confidence":"confidence","Visual quality":"visual_quality"}
        return [{"name":n,"score":round(float(breakdown.get(mapping[n],0) or 0),1),"description":d} for n,d in cls.DIMENSIONS.items()]
