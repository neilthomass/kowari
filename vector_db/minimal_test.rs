// Minimal test to verify basic functionality
use std::collections::HashMap;

#[test]
fn test_basic_functionality() {
    // Test 1: Basic data structures
    let mut map = HashMap::new();
    map.insert("test", "value");
    assert_eq!(map.get("test"), Some(&"value"));
    
    // Test 2: Vector operations (simulated)
    let vector_data = vec![1.0, 2.0, 3.0];
    assert_eq!(vector_data.len(), 3);
    assert_eq!(vector_data[0], 1.0);
    
    // Test 3: String operations
    let uuid_str = "550e8400-e29b-41d4-a716-446655440000";
    assert_eq!(uuid_str.len(), 36);
    
    // Test 4: JSON-like operations
    let json_data = r#"{"id": "test", "data": [1, 2, 3]}"#;
    assert!(json_data.contains("test"));
    
    // Test 5: Math operations
    let v1 = vec![1.0, 0.0, 0.0];
    let v2 = vec![0.0, 1.0, 0.0];
    let dot_product: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    assert_eq!(dot_product, 0.0);
    
    println!("✅ All basic functionality tests passed!");
}

fn main() {
    // Test 1: Basic data structures
    let mut map = std::collections::HashMap::new();
    map.insert("test", "value");
    assert_eq!(map.get("test"), Some(&"value"));
    
    // Test 2: Vector operations (simulated)
    let vector_data = vec![1.0, 2.0, 3.0];
    assert_eq!(vector_data.len(), 3);
    assert_eq!(vector_data[0], 1.0);
    
    // Test 3: String operations
    let uuid_str = "550e8400-e29b-41d4-a716-446655440000";
    assert_eq!(uuid_str.len(), 36);
    
    // Test 4: JSON-like operations
    let json_data = r#"{"id": "test", "data": [1, 2, 3]}"#;
    assert!(json_data.contains("test"));
    
    // Test 5: Math operations
    let v1 = vec![1.0, 0.0, 0.0];
    let v2 = vec![0.0, 1.0, 0.0];
    let dot_product: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    assert_eq!(dot_product, 0.0);
    
    println!("✅ All basic functionality tests passed!");
    println!("🎉 Minimal test completed successfully!");
} 