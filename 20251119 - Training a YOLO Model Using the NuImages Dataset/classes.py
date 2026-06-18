from enum import Enum, auto

# single level categories like YOLO
class NuImageSimpleCategory(Enum):
    pedestrian = 0 # standing, walking
    non_pedestrian = auto() # sitting, lying down
    cyclist = auto() # person on bike
    car = auto() # family car, suv under 9 seats
    large_car = auto() # truck, bus, utility vehicle
    scooter = auto() # standing two-wheeled vehicle
    bicycle = auto() # the bike only no person riding
    motorcyclist = auto() # person on motorcycle
    motorcycle = auto() # motorcycle only

# convert nuImages category and attibute to YOLO-like category
attribute_aware_class_mapping = {
    'animal': None,
    'human.pedestrian.adult': {
        'pedestrian.sitting_lying_down': 'non_pedestrian',
        'pedestrian.moving': 'pedestrian',
        'pedestrian.standing': 'pedestrian',
    },
    'human.pedestrian.child': {
        'pedestrian.sitting_lying_down': 'non_pedestrian',
        'pedestrian.moving': 'pedestrian',
        'pedestrian.standing': 'pedestrian',
    },
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.personal_mobility': 'scooter',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.stroller': None,
    'human.pedestrian.wheelchair': 'non_pedestrian',
    'movable_object.barrier': None,
    'movable_object.pushable_pullable': None,
    'movable_object.debris': None,
    'movable_object.trafficcone': None,
    'static_object.bicycle_rack': None,
    'vehicle.bicycle': {
        'cycle.with_rider': 'cyclist',
        'cycle.without_rider': 'bicycle',
    },
    'vehicle.bus.bendy': 'large_car',
    'vehicle.bus.rigid': 'large_car',
    'vehicle.car': 'car',
    'vehicle.construction': None,
    'vehicle.ego': None,
    'vehicle.emergency.ambulance': 'large_car',
    'vehicle.emergency.police': 'car',
    'vehicle.motorcycle': {
        'cycle.with_rider': 'motorcyclist',
        'cycle.without_rider': 'motorcycle',
    },
    'vehicle.trailer': 'large_car',
    'vehicle.truck': 'large_car'
}

class NuImageSimplerCategory(Enum):
    person = 0
    cyclist = auto()
    car = auto()
    motorcyclist = auto()

# convert nuImages category and attibute to YOLO-like category
simpler_class_mapping = {
    'animal': None,
    'human.pedestrian.adult': {
        'pedestrian.sitting_lying_down': 'person',
        'pedestrian.moving': 'person',
        'pedestrian.standing': 'person',
    },
    'human.pedestrian.child': {
        'pedestrian.sitting_lying_down': 'person',
        'pedestrian.moving': 'person',
        'pedestrian.standing': 'person',
    },
    'human.pedestrian.construction_worker': 'person',
    'human.pedestrian.personal_mobility': 'person',
    'human.pedestrian.police_officer': 'person',
    'human.pedestrian.stroller': None,
    'human.pedestrian.wheelchair': 'person',
    'movable_object.barrier': None,
    'movable_object.pushable_pullable': None,
    'movable_object.debris': None,
    'movable_object.trafficcone': None,
    'static_object.bicycle_rack': None,
    'vehicle.bicycle': {
        'cycle.with_rider': 'cyclist',
        'cycle.without_rider': None,
    },
    'vehicle.bus.bendy': 'car',
    'vehicle.bus.rigid': 'car',
    'vehicle.car': 'car',
    'vehicle.construction': None,
    'vehicle.ego': None,
    'vehicle.emergency.ambulance': 'car',
    'vehicle.emergency.police': 'car',
    'vehicle.motorcycle': {
        'cycle.with_rider': 'motorcyclist',
        'cycle.without_rider': None,
    },
    'vehicle.trailer': 'car',
    'vehicle.truck': 'car'
}

# the simplest YOLO categories
class NuImageSimplestCategory(Enum):
    person = 0 # people
    vehicle = 1 # people on things that make them move

super_simple_class_mapping = {
    'animal': None,
    'human.pedestrian.adult': {
        'pedestrian.sitting_lying_down': 'person',
        'pedestrian.moving': 'person',
        'pedestrian.standing': 'person',
    },
    'human.pedestrian.child': {
        'pedestrian.sitting_lying_down': 'person',
        'pedestrian.moving': 'person',
        'pedestrian.standing': 'person',
    },
    'human.pedestrian.construction_worker': 'person',
    'human.pedestrian.personal_mobility': 'person',
    'human.pedestrian.police_officer': 'person',
    'human.pedestrian.stroller': None,
    'human.pedestrian.wheelchair': 'person',
    'movable_object.barrier': None,
    'movable_object.pushable_pullable': None,
    'movable_object.debris': None,
    'movable_object.trafficcone': None,
    'static_object.bicycle_rack': None,
    'vehicle.bicycle': {
        'cycle.with_rider': 'vehicle',
        'cycle.without_rider': None,
    },
    'vehicle.bus.bendy': 'vehicle',
    'vehicle.bus.rigid': 'vehicle',
    'vehicle.car': 'vehicle',
    'vehicle.construction': None,
    'vehicle.ego': None,
    'vehicle.emergency.ambulance': 'vehicle',
    'vehicle.emergency.police': 'vehicle',
    'vehicle.motorcycle': {
        'cycle.with_rider': 'vehicle',
        'cycle.without_rider': None,
    },
    'vehicle.trailer': 'vehicle',
    'vehicle.truck': 'vehicle'
}

class LabelMappingTypes(Enum):
    FAITHFUL = 0
    SIMPLER = auto()
    SIMPLEST = auto()

def simplify_nuimage_labels(category, attribute, label_mapping_type):
    '''
    Convert nuImage label categories into flat labels for YOLO

    Args:
        label_mapping_type (str): "FAITHFUL", "SIMPLER", or "SIMPLEST", corresponding to how we group nuImages categories
    '''
    assert category in attribute_aware_class_mapping, f"Category: {category} not found in mapping"

    match label_mapping_type:
        case LabelMappingTypes.FAITHFUL.name:
            mapp = attribute_aware_class_mapping[category]
        case LabelMappingTypes.SIMPLER.name:
            mapp = simpler_class_mapping[category]
        case LabelMappingTypes.SIMPLEST.name:
            mapp = super_simple_class_mapping[category]

    if mapp is None:
        # ignore label
        return None
    elif isinstance(mapp, str):
        # 1-to-1 mapping
        return mapp
    elif attribute in mapp:
        return mapp[attribute]
    else:
        return None