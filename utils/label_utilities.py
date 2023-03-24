
import habitat_sim

def isntance_to_semantic(semantic_scene: habitat_sim.scene.SemanticScene) -> None:
    
    instance_ids = [int(obj.id.split("_")[-1]) for obj in semantic_scene.objects]
    category_ids = [obj.category.index() for obj in semantic_scene.objects]
        
    instanceID_to_SemanticID = {instance_id: semantic_id for instance_id, semantic_id in zip(instance_ids, category_ids)}

    return instanceID_to_SemanticID

def id_to_name(semantic_scene: habitat_sim.scene.SemanticScene) -> None:
    
    instance_ids = [int(obj.id.split("_")[-1]) for obj in semantic_scene.objects]
    category_names = [obj.category.name() for obj in semantic_scene.objects]
    
    semanticID_to_label_name = {semantic_id: label_name for semantic_id, label_name in zip(instance_ids, category_names)}
    
    return semanticID_to_label_name